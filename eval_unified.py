#!/usr/bin/env python
"""
Unified evaluation script for Telegrapher, Flow Matching, and Diffusion models.
Computes FID and optionally performs L2 nearest neighbor analysis.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torchdiffeq import odeint
import torch_fidelity
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from utils.unet_oai import UNetModel
from utils.sample_kac import TorchKacConstantSampler


def get_args():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for Telegrapher/FM/Diffusion models"
    )

    # Model selection
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['kac', 'fm', 'diffusion'],
                        help='Model type: kac (telegrapher), fm (flow matching), diffusion')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to EMA checkpoint (.pt file)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (auto-generated if not specified)')

    # Kac parameters
    parser.add_argument('--a', type=float, default=9.0, help='Kac jump rate')
    parser.add_argument('--c', type=float, default=3.0, help='Kac jump size')

    # FM/Diffusion parameters
    parser.add_argument('--sigma', type=float, default=1.0, help='Noise scale')

    # Common parameters
    parser.add_argument('--T', type=float, default=1.0, help='Time horizon')

    # Sampling settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--fid_size', type=int, default=75)
    parser.add_argument('--batch_eval', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--max_batch', type=int, default=100,
                        help='Max batch size for ODE integration (GPU memory)')
    parser.add_argument('--gen_batch', type=int, default=1000,
                        help='Batch size for generation loop')

    # Optional NN analysis
    parser.add_argument('--compute_nn', action='store_true',
                        help='Compute L2 nearest neighbor analysis')
    parser.add_argument('--nn_samples', type=int, default=16,
                        help='Number of samples for NN visualization')
    parser.add_argument('--diff_scale', type=float, default=5.0,
                        help='Scale factor for difference visualization')

    return parser.parse_args()


class Uint8Dataset(Dataset):
    """Wrap uint8 tensor for torch_fidelity FID computation."""
    def __init__(self, tensor_uint8):
        self.data = tensor_uint8
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        return self.data[idx]


def to_uint8_rgb(imgs: torch.Tensor, size: int) -> torch.Tensor:
    """Convert [-1,1] float images to uint8 RGB at specified size."""
    imgs = (imgs + 1) * 0.5
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    imgs = F.interpolate(imgs, size=(size, size), mode='bilinear', align_corners=False)
    return (imgs * 255).round().clamp(0, 255).to(torch.uint8)


class ODEWrapper(torch.nn.Module):
    """Wrap UNet model for torchdiffeq ODE solver."""
    def __init__(self, model, img_size):
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, t, x):
        B = x.shape[0]
        x_img = x.view(B, 3, self.img_size, self.img_size)
        t_vec = torch.full((B,), float(t), device=x.device)
        v = self.model(x_img, t_vec)
        return v.view(x.shape)


def create_initial_noise(model_type, num_samples, dim, args, device):
    """Create initial noise x_T based on model type."""
    if model_type == 'kac':
        sampler = TorchKacConstantSampler(
            a=args.a, c=args.c, T=args.T,
            M=int(50000 * args.T), K=4096
        )
        t_T = torch.ones(num_samples, device='cpu') * args.T
        x_T = sampler.sample(t_T, dim=dim)
        return x_T
    else:
        # FM and Diffusion use Gaussian noise
        eps = torch.randn(num_samples, dim)
        x_T = args.sigma * (args.T ** 0.5) * eps
        return x_T


def generate_samples(model, x_T_cpu, args, device):
    """Generate samples via ODE integration in chunks."""
    ode_fn = ODEWrapper(model, args.img_size).to(device)
    t_vals = torch.linspace(args.T, 1e-5, args.num_steps, device=device)

    x_gen_chunks = []
    total = x_T_cpu.shape[0]

    with torch.no_grad():
        for start in tqdm(range(0, total, args.gen_batch), desc="Generating"):
            end = min(start + args.gen_batch, total)
            x_T_chunk = x_T_cpu[start:end].to(device)

            # ODE solve in sub-batches for memory
            finals = []
            for sub_chunk in x_T_chunk.split(args.max_batch, dim=0):
                sol = odeint(ode_fn, sub_chunk, t_vals, method='euler')
                finals.append(sol[-1])

            x_gen_chunks.append(torch.cat(finals, dim=0).cpu())

            # Memory cleanup
            del x_T_chunk, finals
            torch.cuda.empty_cache()

    return torch.cat(x_gen_chunks, dim=0)


def find_l2_nearest_neighbors(gen_samples, real_dataset, device, batch_size=500):
    """Find L2 nearest neighbor in real dataset for each generated sample."""
    N = gen_samples.shape[0]
    gen_flat = gen_samples.view(N, -1).to(device)

    nn_indices = torch.zeros(N, dtype=torch.long)
    nn_distances = torch.full((N,), float('inf'))

    loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    idx_offset = 0
    for real_batch, _ in tqdm(loader, desc="Finding nearest neighbors"):
        real_batch = real_batch.to(device)
        B = real_batch.shape[0]
        real_flat = real_batch.view(B, -1)

        # Compute pairwise L2 distances
        gen_sq = (gen_flat ** 2).sum(dim=1, keepdim=True)
        real_sq = (real_flat ** 2).sum(dim=1, keepdim=True).T
        dists = torch.sqrt((gen_sq + real_sq - 2 * gen_flat @ real_flat.T).clamp(min=0))

        min_dists, min_idx = dists.min(dim=1)
        min_dists = min_dists.cpu()
        min_idx = min_idx.cpu()

        update_mask = min_dists < nn_distances
        nn_distances[update_mask] = min_dists[update_mask]
        nn_indices[update_mask] = min_idx[update_mask] + idx_offset

        idx_offset += B
        del real_batch, real_flat, dists
        torch.cuda.empty_cache()

    nn_images = torch.stack([real_dataset[i][0] for i in nn_indices])
    return nn_images, nn_indices, nn_distances


def create_nn_visualization(gen_samples, nn_images, diff_scale=5.0):
    """Create visualization: generated | nearest neighbor | scaled difference."""
    diff = (gen_samples - nn_images).abs() * diff_scale
    diff = diff.clamp(-1, 1)
    return torch.cat([gen_samples, nn_images, diff], dim=0)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Auto-generate output directory
    if args.output_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
        args.output_dir = f"eval_{args.model_type}_{ckpt_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model_type} model from {args.checkpoint_path}")
    base_model = UNetModel(
        in_channels=3, out_channels=3,
        num_res_blocks=2, image_size=args.img_size,
        model_channels=128, channel_mult=(1, 2, 2, 2),
        num_heads=4, num_head_channels=64, dropout=0.1,
        attention_resolutions=(16,)
    ).to(device)

    ema_model = AveragedModel(base_model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))
    ema_state = torch.load(args.checkpoint_path, map_location=device)
    ema_model.load_state_dict(ema_state)
    model = ema_model.module if hasattr(ema_model, 'module') else ema_model
    model.eval()

    # Create initial noise
    dim = args.img_size ** 2 * 3
    print(f"Creating initial noise for {args.num_samples} samples...")
    x_T = create_initial_noise(args.model_type, args.num_samples, dim, args, device)

    # Generate samples
    print(f"Generating {args.num_samples} samples with {args.num_steps} Euler steps...")
    x_gen = generate_samples(model, x_T, args, device)
    x_gen = x_gen.view(args.num_samples, 3, args.img_size, args.img_size)

    # Save sample grid
    n_grid = min(args.num_samples, 64)
    save_image(
        x_gen[:n_grid],
        os.path.join(args.output_dir, "gen_grid.png"),
        nrow=8,
        normalize=True
    )

    # Prepare real dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)

    # Optional: L2 Nearest Neighbor Analysis
    if args.compute_nn:
        print(f"\nComputing L2 nearest neighbors for {args.nn_samples} samples...")
        gen_subset = x_gen[:args.nn_samples]
        nn_images, nn_indices, nn_distances = find_l2_nearest_neighbors(
            gen_subset, train_dataset, device, batch_size=500
        )

        nn_grid = create_nn_visualization(gen_subset, nn_images, args.diff_scale)
        save_image(
            nn_grid,
            os.path.join(args.output_dir, 'nn_comparison.png'),
            nrow=args.nn_samples, normalize=True, padding=2, pad_value=1
        )

        with open(os.path.join(args.output_dir, 'nn_stats.txt'), 'w') as f:
            f.write(f"L2 Nearest Neighbor Statistics ({args.nn_samples} samples)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Mean L2 distance: {nn_distances.mean():.4f}\n")
            f.write(f"Min L2 distance:  {nn_distances.min():.4f}\n")
            f.write(f"Max L2 distance:  {nn_distances.max():.4f}\n")
            f.write(f"Std L2 distance:  {nn_distances.std():.4f}\n")

        print(f"Saved NN visualization to {os.path.join(args.output_dir, 'nn_comparison.png')}")
        print(f"Mean L2 distance: {nn_distances.mean():.4f}")

    # FID Computation
    print("\nComputing FID...")
    loader = DataLoader(train_dataset, batch_size=args.batch_eval, shuffle=False)
    real_images = []
    for imgs, _ in loader:
        real_images.append(to_uint8_rgb(imgs, args.fid_size))
    real_images = torch.cat(real_images)

    real_ds = Uint8Dataset(real_images)
    gen_ds = Uint8Dataset(to_uint8_rgb(x_gen, args.fid_size))

    metrics = torch_fidelity.calculate_metrics(
        input1=real_ds,
        input2=gen_ds,
        batch_size=args.batch_eval,
        fid=True,
        cuda=(device == 'cuda'),
        verbose=True
    )
    fid_value = metrics['frechet_inception_distance']

    # Save results
    with open(os.path.join(args.output_dir, 'fid_result.txt'), 'w') as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"FID ({args.num_samples} samples, Euler {args.num_steps} steps): {fid_value:.4f}\n")
        if args.model_type == 'kac':
            f.write(f"Parameters: a={args.a}, c={args.c}, T={args.T}\n")
        else:
            f.write(f"Parameters: sigma={args.sigma}, T={args.T}\n")

    print(f"\nDone. FID: {fid_value:.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
