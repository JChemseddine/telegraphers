#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torchdiffeq import odeint
from torch import optim
import matplotlib.pyplot as plt
import torch_fidelity
import wandb
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import tqdm
from utils.unet_oai import UNetModel
from utils.sample_kac import TorchKacConstantSampler
from utils.velo_utils import compute_velocity
import numpy as np




# -------------------------
# Constants
# -------------------------
_IMG_SIZE = 32

# -------------------------
# Noise schedules
# -------------------------
def get_f(t: torch.Tensor, T: float, name: str) -> torch.Tensor:
    if name == 'linear':
        return 1 - t / T
    elif name == 'exp':
        return torch.exp(-t)
    elif name == 'cos':
        return torch.cos(0.5 * torch.pi * t / T)
    elif name == 'lambda':
        s, ω = -2, torch.pi / (2 * T)
        A = torch.exp(2 / s)
        return (A * torch.cos(ω * t)) / (A * torch.cos(ω * t) + torch.sin(ω * t))
    else:
        raise ValueError(f"Unknown schedule: {name}")


def get_df(t: torch.Tensor, T: float, name: str) -> torch.Tensor:
    if name == 'linear':
        return -1.0 / T * torch.ones_like(t)
    elif name == 'exp':
        return -torch.exp(-t)
    elif name == 'cos':
        return -0.5 * torch.pi / T * torch.sin(0.5 * torch.pi * t / T)
    elif name == 'lambda':
        s, ω = -2, torch.pi / (2 * T)
        A = torch.exp(2 / s)
        num = -ω * A * (torch.sin(ω * t)**2 + torch.cos(ω * t)**2)
        denom = (torch.sin(ω * t) + A * torch.cos(ω * t))**2
        return num / denom
    else:
        raise ValueError(f"Unknown schedule: {name}")

# -------------------------
# Argument parsing
# -------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description="Train Diffusion model on CIFAR-10 with configurable schedules."
    )
    # Training settings
    parser.add_argument('--epochs',          type=int,   default=400_000)
    parser.add_argument('--batch_size',      type=int,   default=128)
    parser.add_argument('--lr',              type=float, default=2e-4)
    parser.add_argument('--seed',            type=int,   default=0)
    # Kac process params
    parser.add_argument('--sigma',               type=float, default=1.0, help="sigma")
    parser.add_argument('--T',               type=float, default=1.0,    help="Time horizon T")
    # Noise schedule
    parser.add_argument('--schedule',        type=str,   default='linear',
                        choices=['linear','exp','cos','lambda'],
                        help="Forward noise schedule.")
    # Evaluation / logging
    parser.add_argument('--eval_interval', type=int, default=5_000)
    parser.add_argument('--eval_interval_fid', type=int, default=10_000)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--fid_num_real', type=int, default=2_000, help="Number of real images for periodic FID eval.")
    parser.add_argument('--final_fid_num_samples', type=int, default=50_000, help="Number of samples for final FID evaluation.")
    parser.add_argument('--fid_image_size', type=int, default=75)
    parser.add_argument('--save_dir', type=str, default='results_diff_cifar10')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    return parser.parse_args()


# -------------------------
# Data / FID Utilities
# -------------------------
class Uint8Dataset(Dataset):
    """Wrap a uint8 tensor for FID evaluation."""
    def __init__(self, tensor_uint8):
        self.data = tensor_uint8
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]


def to_uint8_rgb(imgs: torch.Tensor, size: int) -> torch.Tensor:
    """Convert [-1,1] float to resized uint8 RGB."""
    imgs = (imgs + 1) * 0.5
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1,3,1,1)
    imgs = F.interpolate(imgs, size=(size,size), mode='bilinear', align_corners=False)
    return (imgs*255).round().clamp(0,255).to(torch.uint8)


def calculate_fid(real: torch.Tensor, gen: torch.Tensor, img_size: int, device: str) -> float:
    # make sure inputs are on CPU
    real_cpu = real.detach().cpu()
    gen_cpu  = to_uint8_rgb(gen, img_size).detach().cpu()

    real_ds = Uint8Dataset(real_cpu)
    gen_ds  = Uint8Dataset(gen_cpu)

    metrics = torch_fidelity.calculate_metrics(
        input1=real_ds,
        input2=gen_ds,
        batch_size=256,
        fid=True,
        cuda=(device == 'cuda'),
        verbose=False,
    )
    return metrics['frechet_inception_distance']



def get_unet(img_size: int, channels: int):
    return UNetModel(
        in_channels=channels, out_channels=channels,
        num_res_blocks=2, image_size=img_size,
        model_channels=128, channel_mult=(1,2,2,2),
        num_heads=4, num_head_channels=64,dropout=0.1,
        attention_resolutions=(16,)
    )


class ODEWrapper(torch.nn.Module):
    """Wrap model for torchdiffeq ODE solver, expanding t to a batch vector."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, t, x):
        B = x.shape[0]
        x_img = x.view(B, 3, _IMG_SIZE, _IMG_SIZE)
        # Expand scalar t to a vector for batch
        t_vec = torch.full((B,), float(t), device=x.device)
        v = self.model(x_img, t_vec)
        return v.view(x.shape)


def sample_ode(model, x_T, T, num_steps, device, method='euler', max_batch=500):
    ode_fn = ODEWrapper(model).to(device)
    t_vals = torch.linspace(T,1e-5,num_steps,device=device)
    traj_chunks = []
    with torch.no_grad():
        for chunk in x_T.split(max_batch,dim=0):
            sol = odeint(ode_fn, chunk, t_vals, method=method)
            traj_chunks.append(sol)
    return torch.cat(traj_chunks,dim=1)

# -------------------------
# Main training
# -------------------------
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    
    if args.use_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            mode='online'
        )
    sigma=args.sigma
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Data loaders: CIFAR-10 [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3,(0.5,)*3),
    ])
    train_ds = datasets.CIFAR10('./data',train=True,download=True,transform=transform)
    loader   = DataLoader(train_ds,batch_size=args.batch_size,
                          shuffle=True,num_workers=1,drop_last=True)
    fid_loader = DataLoader(train_ds,batch_size=args.batch_size*4,
                            shuffle=False,num_workers=1)

    device = args.device
    dim = _IMG_SIZE*_IMG_SIZE*3

    # Precompute real images for FID
    real_images = []
    for imgs,_ in fid_loader:
        real_images.append(to_uint8_rgb(imgs.to(device), args.fid_image_size))
        if sum(x.size(0) for x in real_images)>=args.fid_num_real:
            break
    real_images = torch.cat(real_images)[:args.fid_num_real].cpu()
    real_ds = Uint8Dataset(real_images)
    real_stats = torch_fidelity.calculate_metrics(
        input1=real_ds,
        input2=real_ds,
        batch_size=256,
        fid=True,
        cuda=(device == 'cuda'),
        verbose=False,
        stats=True
    )
    # Model, EMA, optimizer, sampler
    unet = get_unet(_IMG_SIZE,3).to(device)
    ema  = AveragedModel(unet, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
    opt  = optim.Adam(unet.parameters(),lr=args.lr)
    # Prepare fixed noise
    with torch.no_grad():
        eps_vis = torch.randn(args.num_samples, dim, device=device)
        xT_fixed_vis = sigma * torch.sqrt(torch.tensor(args.T, device=device)) * eps_vis

        
        eps_fid = torch.randn(args.fid_num_real, dim, device=device)
        xT_fixed_fid = sigma * torch.sqrt(torch.tensor(args.T, device=device)) * eps_fid
    train_iter = iter(loader)
    progress = tqdm(range(args.epochs), desc="Training")
    for step in progress:

        try:
            imgs,_ = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            imgs,_ = next(train_iter)
        x0 = imgs.view(imgs.size(0),-1).to(device)
        B = x0.size(0)

        # Forward process
        t   = torch.rand(B,1,device=device)*args.T
        t= t.clamp(min=1e-5)
        eps = torch.randn_like(x0)
        xt = (1-t) * x0 + torch.sqrt(t) * eps

        velocity = (xt - get_f(t.squeeze(1), args.T, args.schedule).unsqueeze(1) * x0) / ( 2*t)
        drift    = get_df(t.squeeze(1), args.T, args.schedule).unsqueeze(1) * x0

     
        target =velocity + drift
      
        xt_imgs = xt.view(B,3,_IMG_SIZE,_IMG_SIZE)
        pred = unet(xt.view(B, 3, _IMG_SIZE, _IMG_SIZE), t.squeeze(1)).view(B, -1)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(),1.0)
        opt.step(); ema.update_parameters(unet)
        progress.set_description(f"Step {step+1} Loss={loss.item():.5f}")
     # Visualization (Euler only)
        if (step + 1) % args.eval_interval == 0:
            ema.eval()
            # Euler sampling for visualization using num_samples
            traj_e = sample_ode(
                ema, xT_fixed_vis, args.T, 100, device, method='euler'
            )
            x_e = traj_e[-1].view(-1, 3, _IMG_SIZE, _IMG_SIZE).cpu()
            grid_e = make_grid(x_e[:64], nrow=8, normalize=True, scale_each=True)

   
            cpu_e = grid_e.permute(1, 2, 0).cpu().numpy()
            fig = plt.figure(figsize=(6,6))
            plt.imshow(cpu_e)
            plt.axis('off')
            path_e = os.path.join(save_dir, f'samples_euler_{step+1}.png')
            fig.savefig(path_e, bbox_inches='tight')
            wandb.log({"samples_euler": wandb.Image(fig)}, step=step+1)
            plt.close(fig)

        # FID & checkpoints
        if (step + 1) % args.eval_interval_fid == 0:
            ema.eval()

            
            traj_e = sample_ode(
                ema, xT_fixed_fid, args.T, 100, device, method='euler'
            )
            fakes_e = traj_e[-1].view(
                args.fid_num_real, 3, _IMG_SIZE, _IMG_SIZE
            ).cpu()

            gen_ds_e = Uint8Dataset(to_uint8_rgb(fakes_e, args.fid_image_size).cpu())
            metrics_e = torch_fidelity.calculate_metrics(
                input1=real_ds,             
                statistics_real=real_stats, 
                input2=gen_ds_e,            
                batch_size=256,
                fid=True,
                cuda=(device == 'cuda'),
                verbose=False,
            )
            fid_e = metrics_e['frechet_inception_distance']

            wandb.log({"fid_euler": fid_e}, step=step+1)

  
            grid_e_fid = make_grid(fakes_e[:64], nrow=8, normalize=True, scale_each=True)
            cpu_grid_e = grid_e_fid.permute(1, 2, 0).cpu().numpy()
            fig_e = plt.figure(figsize=(6,6))
            plt.imshow(cpu_grid_e)
            plt.axis('off')
            fig_e.savefig(os.path.join(save_dir, f'samples_euler_fid_{step+1}.png'), bbox_inches='tight')
            wandb.log({"samples_euler_fid": wandb.Image(fig_e)}, step=step+1)
            plt.close(fig_e)

            # 6) Save model checkpoints
            torch.save(ema.state_dict(),  os.path.join(save_dir, f'ema_{step+1}.pt'))
            torch.save(unet.state_dict(), os.path.join(save_dir, f'unet_{step+1}.pt'))

    print("Training complete.")
    torch.save(ema.state_dict(), os.path.join(save_dir, 'ema_final.pt'))
    torch.save(unet.state_dict(), os.path.join(save_dir, 'unet_final.pt'))

    # --- Final FID Evaluation on 50k samples ---
    print(f"\nStarting final FID evaluation with {args.final_fid_num_samples} samples...")
    torch.manual_seed(42) # Set new random seed for final evaluation
    ema.eval()

    # 1. Prepare the full 50k real dataset (CIFAR-10 training set)
    print("Loading all real images for final FID...")
    num_final_real = len(train_ds) if len(train_ds) <= args.final_fid_num_samples else args.final_fid_num_samples
    all_real_images = []
    full_loader = DataLoader(train_ds, batch_size=512, shuffle=False)
    for imgs, _ in tqdm(full_loader, desc="Converting real images to uint8"):
        all_real_images.append(to_uint8_rgb(imgs, args.fid_image_size))
        # No need to stop early – we'll truncate below.
    all_real_images_tensor = torch.cat(all_real_images)[:num_final_real]
    real_ds_final = Uint8Dataset(all_real_images_tensor.cpu())

    # 2. Generate 50k fake samples
    print(f"Generating {args.final_fid_num_samples} fake images...")
    t_final = torch.ones(args.final_fid_num_samples, 1, device=device) * args.T
    # Sample latent noise x_T ~ N(0, sigma^2 * T I)
    xT_final = args.sigma * torch.sqrt(torch.tensor(args.T, device=device)) * torch.randn(args.final_fid_num_samples, dim, device=device)
    traj_final = sample_ode(ema, xT_final, args.T, 100, device, method='euler')
    final_fakes = traj_final[-1].view(args.final_fid_num_samples, 3, _IMG_SIZE, _IMG_SIZE).cpu()
    gen_ds_final = Uint8Dataset(to_uint8_rgb(final_fakes, args.fid_image_size).cpu())

    # 3. Calculate final FID
    print("Calculating final FID score...")
    final_metrics = torch_fidelity.calculate_metrics(
        input1=real_ds_final,
        input2=gen_ds_final,
        batch_size=128,
        fid=True,
        cuda=(device == 'cuda'),
        verbose=True,
    )
    final_fid = final_metrics['frechet_inception_distance']
    print("\n-------------------------------------------")
    print(f"Final FID (Euler, {args.final_fid_num_samples} samples, seed 42): {final_fid:.4f}")
    print("-------------------------------------------")
    
    if args.use_wandb:
        wandb.log({"final_fid_euler_50k": final_fid})
        wandb.finish()


if __name__ == '__main__':
    main()
