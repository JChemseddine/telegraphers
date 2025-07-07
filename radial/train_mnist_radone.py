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

from unet_oai import UNetModel
from kac_utils_radial import create_sk, sample_trj
from velo_utils_radial import compute_velocity_nd
from ode_utils import ODEWrapper, TorchWrapper
from geomloss import SamplesLoss
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn



# --------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train radial Kac velocity model on MNIST with FID and sample visualization.")
    parser.add_argument('--epochs', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_kac', type=float, default=1e-4)
    parser.add_argument('--a', type=float, default=9.)
    parser.add_argument('--c', type=float, default=3.)
    parser.add_argument('--T', type=float, default=5.0)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--fid_num_real', type=int, default=2000)
    parser.add_argument('--fid_image_size', type=int, default=75)
    parser.add_argument('--eval_interval', type=int, default=1000)
    
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--ema', type=float, default=0.99)
    parser.add_argument('--f', type=str, default='standart')
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--unet_size', type=str, default='small')

    parser.add_argument('--save_dir', type=str, default='results_radial_mnist')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb', type=bool, default=True)
    return parser.parse_args()

# --------------------------------------------------
def prepare_for_fid(imgs, size):
    imgs = (imgs + 1) / 2.0
    imgs = imgs.repeat(1,3,1,1)
    return torch.nn.functional.interpolate(imgs, size=(size,size), mode='bilinear', align_corners=False)

# --------------------------------------------------
def to_uint8_rgb(imgs, size):
    imgs = (imgs + 1) / 2.0                               # [0,1]
    if imgs.shape[1] == 1:                                # only if grey
        imgs = imgs.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(size, size), mode='bilinear',
                         align_corners=False)
    return (imgs * 255).round().clamp(0, 255).to(torch.uint8)
    
class Uint8Dataset(torch.utils.data.Dataset):
    def __init__(self, tensor_uint8):
        self.t = tensor_uint8
    def __len__(self):
        return self.t.size(0)
    def __getitem__(self, idx):
        return self.t[idx]             # single uint8 Tensor

def calculate_fid(real, gen, args):
    real_ds = Uint8Dataset(to_uint8_rgb(real, args.fid_image_size).cpu())
    gen_ds  = Uint8Dataset(to_uint8_rgb(gen,  args.fid_image_size).cpu())

    metrics = torch_fidelity.calculate_metrics(
        input1      = real_ds,
        input2      = gen_ds,
        batch_size  = 256,             # anything ≥16 is fine
        fid         = True,
        cuda        = (args.device == 'cuda'),
        verbose     = False,
    )
    return metrics['frechet_inception_distance']

# --------------------------------------------------
def get_UNET(img_size=28, channels=1, size = 'small'):
    if size == 'small':
        return UNetModel(
        in_channels=channels,
        out_channels=channels,
        num_res_blocks=2,
        image_size=img_size,
        model_channels=64,
        channel_mult=(1, 2, 4),
        num_heads=1,
        num_head_channels=16,
        attention_resolutions=(256,)
        )
    if size == 'large':
        return UNetModel(
            in_channels=1,
            out_channels=1,
            image_size=28,
            model_channels=128,
            channel_mult=(1, 2, 2, 4),
            num_res_blocks=2,
            attention_resolutions=(7,),  # 28/4 = 7
            num_heads=1,
            num_head_channels=32
        )

class X0Predictor(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = get_UNET(img_size=28, channels=1, size=size)

    def forward(self, x_t, t):
        return self.net(x_t, t)

# --------------------------------------------------
def sampling(fmap, x0_predictor, x_T, T, num_steps, num_samples, mean, variance, dim, device='cuda', max_batch=500):
    ode_func = ODEWrapper(fmap, x0_predictor, T).to(device)
    t_vals = torch.linspace(T, 0., num_steps, device=device)
    chunks = x_T.split(max_batch, dim=0)
    traj_chunks = []
    with torch.no_grad():
        for chunk in chunks:
            x_traj_chunk = odeint(ode_func, chunk, t_vals, method='euler')
            traj_chunks.append(x_traj_chunk)
    x_traj = torch.cat(traj_chunks, dim=1)
    return x_traj

class ODEWrapper(torch.nn.Module):
    def __init__(self, fmap, x0_predictor, T):
        super().__init__()
        self.fmap = fmap
        self.x0_predictor = x0_predictor
        self.T = T

    def forward(self, t, x):
        x_in = x.view(-1, 1, 28, 28)
        t_exp = torch.full((x.shape[0],), t.item(), device=x.device)
        drift = -1.0 / self.T * self.x0_predictor(x_in, t_exp).view(x.shape)
        
        velo = self.fmap(x_in, t_exp).view(x.shape)
        return drift + velo
# --------------------------------------------------
def set_up_wandb(args):
    if args.wandb is False:
        wandb.init(mode="disabled")
   
    wandb.init(
        # set the wandb project where this run will be logged
        project="Kac-Flow-Radial",
        # track hyperparameters and run metadata
        config=vars(args)
    )
    run_name = wandb.run.name
    return run_name

def get_f(t, T, name):
    if name == 'exp':
        return torch.exp(-t)
    elif name == 'cos':
        return torch.cos(0.5 * torch.pi * t/T)
    else: 
        return 1.0 - t / T          
    

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    run_name = set_up_wandb(args)
    save_dir = args.save_dir + '_' + run_name
    os.makedirs(save_dir, exist_ok=True)
    sd = SamplesLoss(blur=1e-3)
    mmd = SamplesLoss('energy')
    device=args.device
    B=args.batch_size
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=args.batch_size, shuffle=True, num_workers=4,drop_last = True)
    fid_loader = DataLoader(mnist, batch_size=args.batch_size*4, shuffle=False, num_workers=4)
    test = DataLoader(mnist, batch_size=args.num_samples, shuffle=True, num_workers=4)
    x_test, _ = next(iter(test))
    x_test= x_test.to(args.device)
    
    # Model & optimizer
    net = get_UNET(img_size=28, channels=1, size=args.unet_size)
        
    net = net.to(args.device)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
    opt = optim.Adam(net.parameters(), lr=args.lr_kac)

    x0_predictor = X0Predictor(size=args.unet_size).to(device)
    ema_x0 = AveragedModel(x0_predictor, multi_avg_fn=get_ema_multi_avg_fn(args.ema))
    optimizer_x0 = optim.Adam(x0_predictor.parameters(), lr=args.lr)


    use_scheduler = False
    if args.scheduler == 'cos':
        use_scheduler == True
        schedular_kac = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30_000, eta_min=args.lr_kac*.1)
        schedular_x0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_x0, T_max=30_000, eta_min=args.lr*.1)

    if args.warm_up > 0:
        warm_up_kac = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=args.lr_kac, total_iters=args.warm_up, )
        warm_up_x0 = torch.optim.lr_scheduler.LinearLR(optimizer_x0, start_factor=0.01, end_factor=args.lr, total_iters=args.warm_up, )
    # Precompute real images for FID
    real_images = []
    for imgs, _ in fid_loader:
        real_images.append(imgs.to(args.device).cpu())
        if len(torch.cat(real_images)) >= args.fid_num_real:
            break
    real_images = torch.cat(real_images)[:args.fid_num_real]

    # Loss setup
    sink = SamplesLoss("energy", p=2, blur=1)
    dim = 28 * 28

    best_fid = float('inf')
    fid_history = []
    step_history = []

    # Infinite iterator over loader
    train_iter = iter(loader)


    progress_bar = tqdm(range(args.epochs), total=args.epochs)
    for step in progress_bar:
        try:
            x0, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            x0, _ = next(train_iter)

        net.train()
        x0_predictor.train()

        t      = torch.rand(B,1, device=device) * args.T       # (B,1)
        t_vec  = t.squeeze(1)  
        x0 = x0.to(device).view(-1,1,28,28)
        x0_flat = x0.view(B, dim)                                # (B,dim)
        
        f       = get_f(t, args.T, args.f)                   # (B,)
        drift   = (-1.0/args.T) * x0_flat                        # (B,dim)

        # sample Kac jump exactly as before
        s_k, sums = create_sk(args.a, t_vec, args.T)
        tau       = sample_trj(t_vec, s_k, sums, args.a, args.T)   # (B,)
        u         = torch.randn(B, dim, device=device)
        u        /= u.norm(dim=1, keepdim=True)
        noise_disp = (args.c * tau).unsqueeze(1) * u               # (B,dim)

        # combine decay + noise
        xt_flat = f * x0_flat + noise_disp            # (B,dim)
        xt      = xt_flat.view(B,1,28,28)
        
        # Optimize x0 net
        x0_hat = x0_predictor(xt, t.squeeze(1)).view(B, -1)
        loss_x0 = F.mse_loss(x0_hat, x0_flat)
        loss_x0.backward()
        optimizer_x0.step()
        optimizer_x0.zero_grad()

        # predict and compute true velocity target
        out_t = net(xt, t_vec).view(B, dim)
        with torch.no_grad():
            # Kac‐part velocity on the noise‐displacement

            disp       = xt_flat - f * x0_flat          # remove the decayed signal
            velo_kac = compute_velocity_nd(
                disp,#noise_disp,
                t_vec.unsqueeze(1),
                args.a, args.c, args.eps
            )
            # total velocity = decay drift + Kac drift
            true_velo = velo_kac

        loss = F.mse_loss(out_t, true_velo)
        opt.zero_grad()
        loss.backward()
        wandb.log({'loss kac': loss.item(), 'loss x0': loss_x0.item()})
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        
        ema.update_parameters(net)
        ema_x0.update_parameters(x0_predictor)

        if use_scheduler is True and step > args.warm_up:
            schedular_kac.step()
            schedular_x0.step()
        if step < args.warm_up:
            warm_up_kac.step()
            warm_up_x0.step()

        progress_bar.set_description(f"Loss={loss.item():.5f} | x0Loss={loss_x0.item():.5f}")

        # periodic evaluation
        if (step+1) % args.eval_interval == 0:
            with torch.no_grad():
                net.eval()
                x0_predictor.eval()
                # sample from prior at t=T
                t_T = torch.full((args.num_samples,1), args.T, device=args.device)
                t_vec_T = t_T.squeeze(1)
                s_k_T, sums_T = create_sk(args.a, t_vec_T, args.T)
                tau_T = sample_trj(t_vec_T, s_k_T, sums_T, args.a, args.T)
                u_T = torch.randn(args.num_samples, dim, device=args.device)
                u_T = u_T / u_T.norm(dim=1, keepdim=True)
                x_T_flat = (args.c * tau_T).unsqueeze(1) * u_T
                x_T =  x_T_flat.view(args.num_samples,dim) #+x_test.view(args.num_samples,dim)

                # reverse integrate
                traj = sampling(net, x0_predictor, x_T, args.T, 100, args.num_samples, 0, args.c**2/args.a * args.T, 28*28)

                x_gen = traj[-1].view(-1,1,28,28)

                # plot and save sample grid
                samples = x_gen[:64].cpu()
                grid = make_grid(samples, nrow=8, normalize=True, scale_each=True)
                plt.figure(figsize=(8,8))
                plt.imshow(grid.permute(1,2,0).squeeze(), cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                sample_path = os.path.join(save_dir, f'samples_epoch_{step}.png')
                plt.savefig(sample_path)
                wandb.log({"samples": wandb.Image(plt)})

                plt.close()
                print(f"Saved sample grid to {sample_path}")
                
                # compute FID
                

                if (step+1) % (args.eval_interval * 10) == 0:
                    t_T = torch.ones((args.fid_num_real,1), device=args.device) * args.T
                    t_vec_T = t_T.squeeze(1)
                    s_k_T, sums_T = create_sk(args.a, t_vec_T, args.T)
                    tau_T = sample_trj(t_vec_T, s_k_T, sums_T, args.a, args.T)
                    u_T = torch.randn(args.fid_num_real, dim, device=args.device)
                    u_T = u_T / u_T.norm(dim=1, keepdim=True)
                    x_T_flat = (args.c * tau_T).unsqueeze(1) * u_T
                    x_T =  x_T_flat.view(args.fid_num_real, dim) #+x_test.view(args.num_samples,dim)

                    traj = sampling(net, x0_predictor, x_T, args.T, 100, args.fid_num_real, 0,
                                    args.c**2/args.a*args.T, dim)
                    x_gen = traj[-1].view(args.fid_num_real, 1, 28, 28).detach()

                    sd_loss = sd(x_gen.reshape(args.fid_num_real, -1), real_images.reshape(args.fid_num_real, -1).to(device))
                    mmd_loss = mmd(x_gen.reshape(args.fid_num_real, -1), real_images.reshape(args.fid_num_real, -1).to(device))
                    wandb.log({'SD':sd_loss, 'MMD': mmd_loss})
                    fid = calculate_fid(real_images, x_gen, args)
                    wandb.log({'fid':fid})
                """fid_history.append(fid)
                step_history.append(epoch)
                print(f"Epoch {epoch}: FID={fid:.4f}")"""
                
                """# save best model
                if fid < best_fid:
                    best_fid = fid
                    torch.save(net.state_dict(), os.path.join(save_dir, 'best_model.pt'))
                    print(f"New best model at epoch {epoch}, FID={fid:.4f}")"""
    
    # save FID history
    np.save(os.path.join(save_dir, 'fid_history.npy'), np.array([step_history, fid_history]))
    
    # plot FID curve
    plt.figure()
    plt.plot(step_history, fid_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.title('Validation FID over Training')
    plt.grid(True)
    plt.tight_layout()
    fid_curve_path = os.path.join(save_dir, 'fid_curve.png')
    plt.savefig(fid_curve_path)
    print(f"Saved FID plot to {fid_curve_path}")
    
if __name__ == '__main__':
    main()
