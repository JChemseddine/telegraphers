#!/usr/bin/env python
# train_radial_mnist.py
# -----------------------------------------------------------
#  Radial Kac velocity model on MNIST + optional paired aug
# -----------------------------------------------------------

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchdiffeq import odeint
import torch_fidelity
import wandb


# ----------------- NEW: batched geometric augmentation -----------------
import kornia.augmentation as K

# ----------------- your own modules ------------------------------------
from unet_oai import UNetModel
from kac_utils_radial import create_sk, sample_trj
from velo_utils_radial import compute_velocity_nd
from geomloss import SamplesLoss
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import net_scheduler as net_s 
# -----------------------------------------------------------------------

# --------------------------------------------------
#  Argument parser
# --------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Train radial Kac velocity model on MNIST with FID and sample visualization."
    )

    # core hyper-params
    parser.add_argument('--epochs', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr_kac', type=float, default=1e-4)

    # process / model
    parser.add_argument('--a', type=float, default=9.)       # jump-rate
    parser.add_argument('--c', type=float, default=3.)       # jump size
    parser.add_argument('--T', type=float, default=5.0)      # horizon
    parser.add_argument('--t_dist', type=str, default='unif') 
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--f', type=str, default='standart')

    parser.add_argument('--unet_size', type=str, default='small',
                        choices=['small', 'large'])
    parser.add_argument(
        "--dropout", type=float, default=0.0, metavar="D",
        help="Dropout rate (0 ≤ D ≤ 1)."
    )

    # eval / bookkeeping
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--fid_num_real', type=int, default=2000)
    parser.add_argument('--fid_image_size', type=int, default=28)

    # optimisation niceties
    parser.add_argument('--scheduler', type=str, default='None',
                        choices=['None', 'cos'])
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--ema', type=float, default=0.99)

    # augmentation toggle  -----------------------------
    parser.add_argument('--paired_aug', action='store_true',
                        help='apply shared RandomAffine + Elastic deformations '
                             'to (x0, xt) each iteration')

    # misc
    parser.add_argument('--save_dir', type=str, default='results_radial_mnist')
    parser.add_argument('--device', type=str,
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', type=bool, default=True)
    return parser.parse_args()

# --------------------------------------------------
#  Utils
# --------------------------------------------------

# You can reuse `to_uint8_rgb` and `Uint8Dataset` from your FID script.
# --------------------------------------------------------------------
# Example:
#   real_ds = Uint8Dataset(to_uint8_rgb(real_imgs, 224).cpu())
#   gen_ds  = Uint8Dataset(to_uint8_rgb(gen_imgs, 224).cpu())
#   cmmd = calculate_cmmd(real_ds, gen_ds, device="cuda")
# --------------------------------------------------------------------

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _preprocess_uint8(batch: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert a uint8 [0‥255] tensor to CLIP-normalised float32."""
    batch = batch.float().div(255.0)                          # [0,1]
    batch = F.interpolate(batch, size=(img_size, img_size),
                         mode="bilinear", align_corners=False)
    mean = torch.tensor(_CLIP_MEAN, device=batch.device).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _get_clip_model(model_name: str = "ViT-B-32", device: str = "cpu"):
    """Load a vision-only CLIP encoder from `open_clip`."""
    try:
        import open_clip
    except ImportError as e:
        raise ImportError("Please `pip install open_clip_torch` to use CMMD.") from e

    model, _, _ = open_clip.create_model_and_transforms(model_name,
                                                        pretrained="openai",
                                                        device=device)
    model.requires_grad_(False)
    model.eval()
    return model


def _encode_with_clip(dataset: torch.utils.data.Dataset,
                      batch_size: int,
                      img_size: int,
                      model_name: str,
                      device: str) -> torch.Tensor:
    """Return L2-normalised CLIP features stacked into one tensor."""
    model = _get_clip_model(model_name, device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"))
    feats = []
    with torch.no_grad():
        for uint8 in loader:                      # uint8 BCHW on CPU
            uint8 = uint8.to(device, non_blocking=True)
            x = _preprocess_uint8(uint8, img_size)  # float BCHW on CUDA/CPU
            f = model.encode_image(x)               # B × D
            f = f / f.norm(dim=-1, keepdim=True)    # L2 normalise
            feats.append(f)
    return torch.cat(feats, dim=0).float()          # N × D (on device)


def _polynomial_mmd(f_real: torch.Tensor, f_gen: torch.Tensor,
                    degree: int = 3) -> float:
    """Unbiased estimate of MMD² with a degree-`degree` polynomial kernel."""
    d = f_real.shape[1]
    scale = 1.0 / d

    def k(a, b):
        return ((scale * a @ b.T) + 1.0).pow(degree)

    k_xx = k(f_real, f_real)
    k_yy = k(f_gen, f_gen)
    k_xy = k(f_real, f_gen)

    # Remove diagonal for unbiased estimate
    m = f_real.size(0)
    n = f_gen.size(0)
    k_xx = k_xx - torch.diag(torch.diag(k_xx))
    k_yy = k_yy - torch.diag(torch.diag(k_yy))

    mmd2 = (k_xx.sum() / (m * (m - 1)) +
            k_yy.sum() / (n * (n - 1)) -
            2 * k_xy.mean())
    return mmd2.item()


def calculate_cmmd(real_ds: torch.utils.data.Dataset,
                   gen_ds: torch.utils.data.Dataset,
                   *,
                   clip_model: str = "ViT-B-32",
                   img_size: int = 224,
                   batch_size: int = 256,
                   degree: int = 3,
                   device: str = "cuda") -> float:
    """Compute CMMD (CLIP-MMD) between *real_ds* and *gen_ds*.

    Args:
        real_ds / gen_ds: `Dataset` that yields uint8 tensors (C×H×W).
        clip_model:  CLIP variant in `open_clip` (e.g. "ViT-L-14").
        img_size:    Side length that CLIP expects (224 or 256).
        batch_size:  Feature extraction mini-batch.
        degree:      Degree of the polynomial kernel (3 ⇒ KID-style).
        device:      "cpu" | "cuda".

    Returns: CMMD² (float).  Smaller ⇒ closer distributions.
    """
    f_real = _encode_with_clip(real_ds, batch_size, img_size, clip_model, device)
    f_gen  = _encode_with_clip(gen_ds,  batch_size, img_size, clip_model, device)
    return _polynomial_mmd(f_real, f_gen, degree)

def to_uint8_rgb(imgs, size):
    """[-1,1] → uint8 RGB and resize for FID."""
    imgs = (imgs + 1) / 2.0
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(size, size), mode='bilinear',
                         align_corners=False)
    return (imgs * 255).round().clamp(0, 255).to(torch.uint8)

class Uint8Dataset(torch.utils.data.Dataset):
    def __init__(self, tensor_uint8):
        self.t = tensor_uint8
    def __len__(self):   return self.t.size(0)
    def __getitem__(self, idx):   return self.t[idx]

def calculate_fid(real, gen, args):
    real_ds = Uint8Dataset(to_uint8_rgb(real, args.fid_image_size).cpu())
    gen_ds  = Uint8Dataset(to_uint8_rgb(gen,  args.fid_image_size).cpu())

    metrics = torch_fidelity.calculate_metrics(
        input1=real_ds,
        input2=gen_ds,
        batch_size=256,
        fid=True,
        cuda=(args.device == 'cuda'),
        verbose=False,
    )
    return metrics['frechet_inception_distance']

def get_UNET(img_size=28, channels=1, size="small", dropout=0.0):
    if size == "small":
        return UNetModel(
            in_channels=channels,
            out_channels=channels,
            num_res_blocks=2,
            image_size=img_size,
            model_channels=64,
            channel_mult=(1, 2, 4),
            num_heads=1,
            num_head_channels=16,
            attention_resolutions=(256,),
            dropout=dropout,          # ← NEW
        )
    # large
    return UNetModel(
        in_channels=channels,
        out_channels=channels,
        image_size=img_size,
        model_channels=96,
        channel_mult=(1, 2, 4),
        num_res_blocks=3,
        attention_resolutions=(4, 2),
        num_heads=3,
        num_head_channels=32,
        dropout=dropout,              # ← NEW
    )
    
class ODEWrapper(torch.nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.fmap = fmap
        

    def forward(self, t, x):
        x_in = x.view(-1, 1, 28, 28)
        t_exp = torch.full((x.shape[0],), t.item(), device=x.device)
        
        return self.fmap(x_in, t_exp).view(x.shape)

                
def sampling(fmap, x_T, T, num_steps, device='cuda', max_batch=500):
    """Euler solve reverse ODE."""
    ode_func = ODEWrapper(fmap).to(device)
    t_vals   = torch.linspace(T, 0., num_steps, device=device)

    chunks = x_T.split(max_batch, dim=0)
    traj_chunks = []
    with torch.no_grad():
        for chunk in chunks:
            x_traj_chunk = odeint(ode_func, chunk, t_vals, method='euler')
            traj_chunks.append(x_traj_chunk)
    return torch.cat(traj_chunks, dim=1)  # (steps,B,dim)

def get_time_dist(name):
    if name == 'unif':
        return lambda batch_size, device: torch.rand(batch_size, 1, device=device)
    elif name == 'beta':
        return lambda batch_size, device: torch.distributions.Beta(0.5, 0.5).sample((batch_size, 1)).to(device)
    else:
        raise NotImplementedError


def get_f(t, T, name, net=None):
    if name == 'exp':
        return torch.exp(-t)
    elif name == 'cos':
        return torch.cos(0.5 * torch.pi * t/T)
    elif name == 'lambda':
        s = -2
        return ((np.exp(2/s) * torch.cos(0.5*torch.pi * t / T)) / (np.exp(2/s) * torch.cos(0.5*torch.pi * t / T) + torch.sin(0.5*torch.pi * t / T)))
    elif name == 'net':
        y_raw = (-net(t.unsqueeze(1) )).flatten()

        # anchor values at the boundaries         (use same device as t)
        y0 =(-net(torch.tensor([0.0], device=t.device) )).item()
        y1 = (-net(torch.tensor([T],    device=t.device))).item()

        # linear transform s.t.  y_raw==y0 → 1  and  y_raw==y1 → 0
        denom = y0 - y1
        #if abs(denom) < 1e-12:                      # avoid divide-by-zero
        #    raise ValueError("y0 and y1 are (almost) equal; cannot rescale.")

        y = (torch.sigmoid((y_raw - y1) / denom  ) - 0.5)  / (torch.sigmoid(torch.tensor([(y0 - y1) / denom], device=t.device) )   -0.5)       
        return y.unsqueeze(1)
    else: 
        return 1.0 - t / T  
    
def get_df(t, T, name, net=None):
    if name == 'linear':
        return -1.0 / T
    elif name == 'exp':
        return -torch.exp(-t) 
    elif name == 'lambda':
        s=-2
        omega = torch.pi / (2.0 * T)          # π / (2 T)
        A = np.exp(2.0 / s)                # e^{2/s}
        # full numerator as written
        num = -torch.pi * A * (torch.sin(omega * t)**2 + torch.cos(omega * t)**2)
        # denominator
        denom = 2.0 * T * (torch.sin(omega * t) + A * torch.cos(omega * t))**2
        return num / denom
    elif name == 'net':
        y_raw = (-net(t.unsqueeze(1) )).flatten()

        # anchor values at the boundaries         (use same device as t)
        y0 =(-net(torch.tensor([0.0], device=t.device) )).item()
        y1 = (-net(torch.tensor([T],    device=t.device))).item()

        # linear transform s.t.  y_raw==y0 → 1  and  y_raw==y1 → 0
        denom = y0 - y1
        #if abs(denom) < 1e-12:                      # avoid divide-by-zero
        #    raise ValueError("y0 and y1 are (almost) equal; cannot rescale.")

        y = (torch.sigmoid((y_raw - y1) / denom  ) - 0.5)  / (torch.sigmoid(torch.tensor([(y0 - y1) / denom], device=t.device) )   -0.5)       
        y = y.unsqueeze(1)
        
        dt = torch.autograd.grad(y , t, grad_outputs=torch.ones_like(y),
                                 retain_graph=True, create_graph=True)[0]
        return dt
    
def plot_f(T, net): 
    T = 10.0  # Example time constant
    with torch.no_grad():
        t = torch.linspace(0, T, 100, device='cuda')
        y_raw = (-net(t.unsqueeze(1) )).flatten()

        # anchor values at the boundaries         (use same device as t)
        y0 =(-net(torch.tensor([0.0], device=t.device) )).item()
        y1 = (-net(torch.tensor([T],    device=t.device))).item()

        # linear transform s.t.  y_raw==y0 → 1  and  y_raw==y1 → 0
        denom = y0 - y1
        #if abs(denom) < 1e-12:                      # avoid divide-by-zero
        #    raise ValueError("y0 and y1 are (almost) equal; cannot rescale.")

        y = (torch.sigmoid((y_raw - y1) / denom  ) - 0.5)  / (torch.sigmoid(torch.tensor([(y0 - y1) / denom], device=t.device) )   -0.5)     

        # optional: keep everything inside [0,1]
        #y = y.clamp(0., 1.)
        plt.figure(figsize=(8, 8))
        plt.plot(t.cpu(),y.cpu())
        wandb.log({"f_schedule": wandb.Image(plt)})
        plt.show()
# --------------------------------------------------
#  Training
# --------------------------------------------------
def main():
    args = get_args()
    torch.manual_seed(args.seed)

    # ----------- logging -------------------
    if args.wandb is False:
        wandb.init(mode="disabled")
    wandb.init(project="Kac-Flow-Radial", config=vars(args))
    run_name = wandb.run.name
    save_dir = f"{args.save_dir}_{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    # ----------- data ----------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    mnist = datasets.MNIST('./data', train=True, download=True,
                           transform=transform)
    loader      = DataLoader(mnist, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, drop_last=True)
    fid_loader  = DataLoader(mnist, batch_size=args.batch_size*4,
                             shuffle=False, num_workers=4)

    device = args.device
    B      = args.batch_size
    dim    = 28 * 28

    # ----------- model & opt ---------------
    net = get_UNET(size=args.unet_size, dropout=args.dropout).to(device)

    
    if args.f == 'net':
        net_scheduler = net_s.NetScheduler(num_hidden_layers=1024).to(device)
    else:
        net_scheduler =  None
    
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters in net: {num_params}")
    
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
    opt = optim.Adam(net.parameters(), lr=args.lr_kac)
    if args.f == 'net':
        opt = optim.Adam([
            { 'params': net.parameters(), 'lr': args.lr_kac },
            { 'params': net_scheduler.parameters(),  'lr': args.lr_kac }],)

    # schedulers
    use_sched = (args.scheduler == 'cos')
    if use_sched:
        sched_kac = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=30_000, eta_min=args.lr_kac * .1
        )
    if args.warm_up > 0:
        warm_kac = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=args.warm_up
        )

    # -------- paired augmentation pipeline -------------
    if args.paired_aug:
        aug_pipeline = torch.nn.Sequential(
            K.RandomAffine(
                degrees=15,
                translate=(2/28, 2/28),
                scale=(1.0, 1.0),     # <- change here (tuple keeps the size fixed)
                # or simply: scale=None
                shear=0.0,
                padding_mode='zeros',
                p=1.0,
            ),
            K.RandomElasticTransform(
                kernel_size=(9, 9),
                sigma=(4.0, 6.0),
                alpha=(30.0, 50.0),
                align_corners=False,
                p=0.5,
            )
        ).to(device).eval()
    else:
        aug_pipeline = None

    # -------- pre-compute real batch for fast FID -------
    real_imgs = []
    for imgs, _ in fid_loader:
        real_imgs.append(imgs.to(device).cpu())
        if len(torch.cat(real_imgs)) >= args.fid_num_real:
            break
    real_imgs = torch.cat(real_imgs)[:args.fid_num_real]

    # --------- training loop ---------------------------
    train_iter = iter(loader)
    progress   = tqdm(range(args.epochs), total=args.epochs)

    plot_f(args.T, net_scheduler)  # initial plot of f(t)

    for step in progress:
        # ---------------- mini-batch -------------------
        try:
            x0, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            x0, _ = next(train_iter)

        x0 = x0.to(device).view(-1, 1, 28, 28)        # (B,1,28,28)

        # ----------------- forward SDE step -----------
        t      = get_time_dist(args.t_dist)(B, device) * args.T#torch.rand(B, 1, device=device) * args.T
        t.requires_grad_(True)  # for net_scheduler
        t_vec  = t.squeeze(1)                          # (B,)
        f_t    = get_f(t, args.T, args.f, net_scheduler)#1.0 - t / args.T                     # linear schedule

        # exact Kac jump
        u  = torch.randn(B, dim, device=device)
        u /= u.norm(dim=1, keepdim=True)
        s_k, sums = create_sk(args.a, t_vec, args.T)
        tau       = sample_trj(t_vec, s_k, sums, args.a, args.T)   # (B,)
        noise_disp = (args.c * tau).unsqueeze(1) * u               # (B,dim)

        x0_flat = x0.view(B, dim)
        xt_flat = f_t * x0_flat + noise_disp
        xt      = xt_flat.view(B, 1, 28, 28)

        # ---------------- paired augmentation ---------
        if aug_pipeline is not None:
            with torch.no_grad():                 # aug = deterministic op
                cat   = torch.cat([x0, xt], dim=0)        # (2B,1,28,28)
                cat   = aug_pipeline(cat)
                x0, xt = cat[:B], cat[B:]
                x0 = torch.clamp(x0, -1.0, 1.0)
                xt = torch.clamp(xt, -1.0, 1.0)

        # refresh flats & drift after possible aug
        x0_flat = x0.view(B, dim)
        xt_flat = xt.view(B, dim)
        drift   = get_df(t, args.T, args.f, net_scheduler) * x0_flat

        # --------------- training step ---------------
        net.train()
        pred_velo = net(xt, t_vec).view(B, dim)

        with torch.no_grad():
            disp = xt_flat - f_t * x0_flat
            velo_kac = compute_velocity_nd(
                disp, t_vec.unsqueeze(1),
                args.a, args.c, args.eps
            )
            
            target_velo = drift + velo_kac

        loss = F.mse_loss(pred_velo, target_velo)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        ema.update_parameters(net)

        # schedulers
        if use_sched and step >= args.warm_up:
            sched_kac.step()
        if args.warm_up and step < args.warm_up:
            warm_kac.step()

        wandb.log({'loss': loss.item()})
        progress.set_description(f"Loss={loss.item():.5f}")

        # ------------- evaluation / visualization ------------
        if (step + 1) % args.eval_interval == 0:
            net.eval()
            ema.eval()
            plot_f(args.T, net_scheduler)
            with torch.no_grad():
                # prior sample at t=T
                tT      = torch.full((args.num_samples, 1), args.T,
                                     device=device)
                tT_vec  = tT.squeeze(1)
                uT      = torch.randn(args.num_samples, dim, device=device)
                uT     /= uT.norm(dim=1, keepdim=True)
                s_kT, sums_T = create_sk(args.a, tT_vec, args.T)
                tauT = sample_trj(tT_vec, s_kT, sums_T, args.a, args.T)
                xT_flat = (args.c * tauT).unsqueeze(1) * uT
                traj = sampling(ema, xT_flat, args.T, num_steps=100,
                                device=device)
                x_gen = traj[-1].view(-1, 1, 28, 28)

                # save sample grid
                grid = make_grid(x_gen[:64].cpu(), nrow=8,
                                 normalize=True, scale_each=True)
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                sample_path = os.path.join(save_dir,
                                           f'samples_step_{step+1}.png')
                plt.savefig(sample_path)
                wandb.log({"samples": wandb.Image(plt)})
                plt.close()

            # optional quick FID every 10 evals
            if (step + 1) % (args.eval_interval * 10) == 0:
                with torch.no_grad():
                    # produce as many fakes as reals for FID
                    tT = torch.ones((args.fid_num_real, 1),
                                    device=device) * args.T
                    tT_vec = tT.squeeze(1)
                    uT = torch.randn(args.fid_num_real, dim, device=device)
                    uT /= uT.norm(dim=1, keepdim=True)
                    s_kT, sums_T = create_sk(args.a, tT_vec, args.T)
                    tauT = sample_trj(tT_vec, s_kT, sums_T,
                                      args.a, args.T)
                    xT_flat = (args.c * tauT).unsqueeze(1) * uT
                    traj = sampling(ema, xT_flat, args.T, 100,
                                    device=device)
                    fakes = traj[-1].view(args.fid_num_real, 1, 28, 28)
                    fid = calculate_fid(real_imgs, fakes, args)
                    #cmmd = calculate_cmmd(real_imgs, fakes)
                    #wandb.log({'fid': fid, 'ccmd': cmmd})
                    wandb.log({'fid': fid,})

    print("Training complete.")


# --------------------------------------------------
if __name__ == '__main__':
    main()