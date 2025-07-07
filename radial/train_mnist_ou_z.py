import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
from torch import optim
from tqdm import tqdm
from torchdiffeq import odeint
from unet_oai import UNetModel
from kac_sampler import *
from velo_utils_final import *
from ode_utils import *
from geomloss import SamplesLoss
import torch_fidelity
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500000)
    parser.add_argument('--inner_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--a', type=float, default=25)
    parser.add_argument('--c', type=float, default=5)
    parser.add_argument('--T', type=float, default=1.)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--fid_num_real_samples', type=int, default=2000)
    parser.add_argument('--fid_image_size', type=int, default=75)
    parser.add_argument('--eps', type=float, default=1e-4, help="Epsilon Boundary")

    return parser.parse_args()

def get_UNET(img_size=28, channels=1):
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

class X0Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = get_UNET(img_size=28, channels=1)

    def forward(self, x_t, t):
        return self.net(x_t, t)
    
def sampling(fmap, x0_predictor, x_T, T, num_steps, num_samples, mean, variance, dim, device=device, max_batch=500):
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

class TensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors.cpu()
    def __len__(self):
        return self.tensors.shape[0]
    def __getitem__(self, idx):
        return self.tensors[idx]

def prepare_images_for_inception(images_tensor, fid_image_size):
    images_tensor = (images_tensor + 1) / 2.0
    images_3ch = images_tensor.repeat(1, 3, 1, 1)
    resize_transform = transforms.Resize((fid_image_size, fid_image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    resized_images = resize_transform(images_3ch)
    return (resized_images * 255).byte()

def calculate_fid(real_images_tensor_cpu, generated_samples_tensor, args):
    prepared_gen_images = prepare_images_for_inception(generated_samples_tensor, args.fid_image_size).cpu()
    real_dataset = TensorDataset(real_images_tensor_cpu[:2000])
    generated_dataset = TensorDataset(prepared_gen_images)
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=real_dataset,
        input2=generated_dataset,
        cuda=(args.device == 'cuda'),
        fid=True,
        verbose=False
    )
    return metrics_dict['frechet_inception_distance']

def get_all_real_images_for_fid(dataloader, num_samples, fid_image_size, device):
    all_real_images = []
    count = 0
    for images, _ in dataloader:
        images = images.to(device)
        prepared_images_batch = prepare_images_for_inception(images, fid_image_size).cpu()
        all_real_images.append(prepared_images_batch)
        count += prepared_images_batch.shape[0]
        if count >= num_samples:
            break
    final_real_images = torch.cat(all_real_images, dim=0)[:num_samples]
    print(f"Collected {final_real_images.shape[0]} real images for FID (as CPU tensor).")
    return final_real_images

def visualize_and_evaluate(fmap, ema_model, x0_predictor, x_T, args, save_dir, train_batch, k, real_images_tensor_for_fid_cpu):
    fid_scores = {}
    for name, model in [('normal', fmap), ('ema', ema_model.module)]:
        fmap_s = TorchWrapper(model)
        traj = sampling(fmap, x0_predictor, x_T, args.T, 100, args.num_samples, 0, args.c**2/args.a * args.T, 28*28)
        transformed_samples = traj[-1].view(-1, 1, 28, 28)
        
        # Save grid of samples as PNG
        grid_img = make_grid(transformed_samples[:16], nrow=8)
        np_grid = grid_img.permute(1, 2, 0).cpu().numpy()
        #np_clipped = np.clip((np_grid + 1)/2, 0, 1)
        plt.figure(figsize=(8, 2))
        plt.imshow(np_grid, cmap="gray")
        plt.axis("off")
        plt.title(f"{name.upper()} FID Samples")
        save_path = os.path.join(save_dir, f'generated_samples_{name}_{k}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        """
        # FID computation
        fid = calculate_fid(real_images_tensor_for_fid_cpu, transformed_samples, args)
        fid_scores[name] = fid
        
    print(f"FID - Normal: {fid_scores['normal']:.6f} | EMA: {fid_scores['ema']:.6f}")
    return fid_scores
    """
def get_kac_weight(t, a, c, w_min=0.1, w_max=10.0, eps=1e-6):
    """
    Weight ∝ 1 / Var[τ_t] ≈ a / (c^2 t)
    """
    raw_weight = a / (c**2 * t + eps)
    return raw_weight.clamp(min=w_min, max=w_max).detach()

def main():
    # SET UP
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 784
    torch.manual_seed(args.seed)
    args.save_dir = f"resultsFID_{args.a}_{args.c}_{args.T}_{args.eps}"
    os.makedirs(args.save_dir, exist_ok=True)

    # LOADERS
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True, num_workers=4 if device == 'cuda' else 0, pin_memory=True if device == 'cuda' else False,drop_last =True)
    fid_real_dataloader = DataLoader(mnist_train, batch_size=args.batch_size * 4, shuffle=False, num_workers=4 if device == 'cuda' else 0, pin_memory=True if device == 'cuda' else False)
    test_loader = DataLoader(mnist_train, batch_size=args.num_samples, shuffle=False, num_workers=4 if device == 'cuda' else 0, pin_memory=True if device == 'cuda' else False)
    real_imgs, big_labels = next(iter(test_loader))

    # SET UP NETWORKS
    print(real_imgs.shape)
    fmap = get_UNET().to(device)
    
    # for Exponential Moving Average 
    ema_model = AveragedModel(fmap, multi_avg_fn=get_ema_multi_avg_fn(0.97))

    optimizer = optim.Adam(fmap.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30_000, eta_min=5e-5)
    x0_predictor = X0Predictor().to(device)
    optimizer_x0 = optim.Adam(x0_predictor.parameters(), lr=args.lr)
    scheduler_x0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_x0, T_max=30_000, eta_min=5e-5)

    sampler = TorchKacZigguratSampler(a=args.a, c=args.c, T=args.T, M=500, n=512)
   
    progress_bar = tqdm(range(args.epochs), total=args.epochs)
    losses = []
    fid_scores_all = []

    real_images_tensor_for_fid = get_all_real_images_for_fid(fid_real_dataloader, args.fid_num_real_samples, args.fid_image_size, device)
    dataloader_iter = iter(train_loader)

    for k in progress_bar:
        optimizer.zero_grad()
        try:
            x0, _ = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            x0, _ = next(dataloader_iter)
       
        x0 = x0.view(x0.size(0), -1).to(device)
        batch_size = x0.shape[0]

        with torch.no_grad():
            t = torch.rand((args.batch_size,1),device=device)*args.T
            tau_t = sampler.sample_kac(t.squeeze(1),dim=dim).to(device)

        x0_flat = x0.view(batch_size, -1)              # (B, dim)

        # 2) Compute fade‐out factor
        f = (1.0 - t / args.T).view(batch_size, 1)     # (B,1)

        # 3) Build new forward sample: decayed x0 + Kac noise
        #    noise_disp is already c * tau_t * u, but if you have raw tau_t:
        noise_disp = tau_t                             # (B, dim)
        xt_flat    = f * x0_flat + noise_disp          # (B, dim)
        xt         = xt_flat.view(batch_size,1,28,28)  # (B,1,28,28)
        x0_hat = x0_predictor(xt, t.squeeze(1)).view(batch_size, -1)
        loss_x0 = F.mse_loss(x0_hat, x0_flat)
        loss_x0.backward()
        optimizer_x0.step()
        optimizer_x0.zero_grad()
        # 4) Network prediction (sees only xt, t)
        out_t = fmap(xt, t.squeeze(1)).view(batch_size, -1)

        # 5) Compute true velocity (only in training) as drift + Kac part
        with torch.no_grad():
            # 5a) Kac drift on the pure-noise displacement
            disp       = xt_flat - f * x0_flat          # remove the decayed signal
            velo_kac   = compute_velocity(
                            disp,
                            t,
                            args.a, args.c, epsilon=args.eps
                        )                              # (B, dim)

            # 5b) Linear drift to wash out x0
            #drift      = (-1.0 / args.T) * x0_flat       # (B, dim)

            # 5c) Total true velocity
            true_vel   = velo_kac               # (B, dim)

        # 6) Loss & backward
        loss = F.mse_loss(out_t, true_vel)
        loss.backward()
        max_norm = 5.0
        torch.nn.utils.clip_grad_norm_(fmap.parameters(), max_norm)
        optimizer.step()
        ema_model.update_parameters(fmap)

        progress_bar.set_description(f"Loss={loss.item():.5f} | x0Loss={loss_x0.item():.5f}")
        losses.append(loss.item())

        scheduler.step()
        scheduler_x0.step()

        if k % 5000 == 0:
           
            torch.save(fmap.state_dict(), os.path.join(args.save_dir, f'fmap_state_dict.pt'))
            torch.save(losses, os.path.join(args.save_dir, 'loss_curve.pt'))
            
            
            N = args.num_samples
            D = 28 * 28
            max_batch = 20

            all_tau = []
            with torch.no_grad():
                

                T_f= torch.ones((N,),  device=device)   *args.T
                tau_T = sampler.sample_kac(T_f,dim=dim).to(device)
                print("VARIANCE:", tau_T.var())

            fid_scores = visualize_and_evaluate(fmap, ema_model, x0_predictor, tau_T, args, args.save_dir, (x0, _), k, real_images_tensor_for_fid)
            #fid_scores_all.append(fid_scores)
            #with open(os.path.join(args.save_dir, 'val_fid.txt'), 'a') as f:
            #    f.write(f"{k}: NormalFID={fid_scores['normal']:.6f}, EMAFID={fid_scores['ema']:.6f}\n")
            #torch.save(fid_scores_all, os.path.join(args.save_dir, 'fid_curve.pt'))

if __name__ == '__main__':
    main()
