import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
from torchdiffeq import odeint
from ode_utils2d import ODEWrapper, TorchWrapper
from action_model import *
from geomloss import SamplesLoss
import itertools


def visualize_and_save(fmap_s, gmm, T, output_dir, num_steps=50,
                       num_samples=5000, dim=2,epsilon =0.0, device='cuda'):
    import os
    from matplotlib import animation
    import numpy as np 
    with torch.no_grad():
        # Sample x_T for VE diffusion
        x0 = gmm.sample((num_samples,)).to(device)
        
        xT = torch.rand_like(x0) * (args.b - args.a) + args.a
        ode_func = ODEWrapper(fmap_s).to(device)

        # Nonlinear time stepping: slow near t=0
        slow_frac = 0.2
        fast_steps = int(num_steps * (1 - slow_frac))
        slow_steps = num_steps - fast_steps
        t_fast = np.linspace(T, T * slow_frac, fast_steps, endpoint=False)
        t_slow = np.linspace(T * slow_frac, epsilon, slow_steps)
        t_vals = torch.tensor(np.concatenate([t_fast, t_slow]),
                              device=device, dtype=torch.float32)

        # Integrate reverse ODE
        x_traj = odeint(ode_func, xT, t_vals, method='dopri5')

    # Setup plotting
    smoothed_xlim = None
    smoothed_ylim = None
    alpha = 0.09
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        nonlocal smoothed_xlim, smoothed_ylim
        ax.clear()
        x_frame = x_traj[frame].cpu().numpy()
        min_vals = x_frame.min(axis=0)
        max_vals = x_frame.max(axis=0)
        padding = 0.2 * (max_vals - min_vals + 1e-3)
        xlim_raw = (min_vals[0] - padding[0], max_vals[0] + padding[0])
        ylim_raw = (min_vals[1] - padding[1], max_vals[1] + padding[1])
        if smoothed_xlim is None:
            smoothed_xlim, smoothed_ylim = xlim_raw, ylim_raw
        else:
            smoothed_xlim = (
                alpha * xlim_raw[0] + (1 - alpha) * smoothed_xlim[0],
                alpha * xlim_raw[1] + (1 - alpha) * smoothed_xlim[1]
            )
            smoothed_ylim = (
                alpha * ylim_raw[0] + (1 - alpha) * smoothed_ylim[0],
                alpha * ylim_raw[1] + (1 - alpha) * smoothed_ylim[1]
            )
        ax.set_xlim(*smoothed_xlim)
        ax.set_ylim(*smoothed_ylim)
        ax.set_title(f"Time t = {t_vals[frame]:.2f}")
        ax.scatter(x_frame[:, 0], x_frame[:, 1], alpha=0.5, s=2)

    # Create animation
    pause = 20
    frames = list(range(num_steps)) + [num_steps - 1] * pause
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
    gif_path = os.path.join(output_dir, 'trajectory.gif')
    ani.save(gif_path, writer='imagemagick', fps=5)
    plt.close(fig)
def plot_trajectories(traj, samples_true, epoch, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    n = min(2000, traj.shape[1])
    plt.figure(figsize=(6,6))
    # target cloud
    plt.scatter(samples_true[:,0], samples_true[:,1], alpha=0.3, c="red", s=5, label='Target')
    # start points z(T)
    plt.scatter(traj[0,:n,0], traj[0,:n,1], s=10, alpha=0.8, c="black", label='z(T)')
    # intermediate
    pts = traj[1:-1,:n,:].reshape(-1,2)
    plt.scatter(pts[:,0], pts[:,1], s=0.5, alpha=0.2, c="olive")
    # final endpoints z(0)
    plt.scatter(traj[-1,:n,0], traj[-1,:n,1], s=4, alpha=1, c="blue", label='z(0)')
    plt.legend(loc='upper right')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    fname = os.path.join(out_dir, f'trajectory_epoch_{epoch}.png')
    plt.savefig(fname, dpi=150)
    plt.close()

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train VE-diffusion velocity model")
    parser.add_argument('--a',        type=float, default=-3.0, help="Constant noise scale σ")
    parser.add_argument('--b',        type=float, default=3.0, help="Constant noise scale σ")
    parser.add_argument('--T',            type=float, default=10.0, help="Time horizon T")
    parser.add_argument('--sigma_gmm', type=float, default=.0001, help="Time horizon T")

    parser.add_argument('--ntrain',       type=int,   default=200000, help="Number of training steps")
    parser.add_argument('--batch_size',   type=int,   default=256,    help="Training batch size")
    parser.add_argument('--val_interval', type=int,   default=20000,  help="Validate every N steps")
    parser.add_argument('--val_samples',  type=int,   default=5000,   help="Samples for validation")
    parser.add_argument('--num_steps_eval', type=int, default=100,   help="ODE steps for validation")
    parser.add_argument('--results_dir',  type=str,   default="results2d_diff", help="Results directory")
    parser.add_argument('--eps',          type=float, default=1e-4,  help="Epsilon for stability")
    parser.add_argument('--seed',         type=int,   default=0,     help="Random seed")
    parser.add_argument('--p', type=float, default=1.5,
                    help="Wasserstein-p gradient-flow parameter (must be > 1 and finite)")

    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.results_dir = f"results_mmd_{args.sigma_gmm}"
    # Setup
    device = 'cuda'

    T = args.T
    dim = 2
    subdir = f"p{args.p}_b{args.b}_a{args.a}_T{T}"
    output_dir = os.path.join(args.results_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    grid_points = list(itertools.product([-1.0, 0.0, 1.0], repeat=dim))
    means = torch.tensor(grid_points, device=device)

    # Uniform weights and fixed small sigma
    weights = torch.full((len(means),), 1.0 / len(means), device=device)
    sigmas = torch.ones_like(means) * args.sigma_gmm  # isotropic

    mix = torch.distributions.Categorical(weights)
    comp = torch.distributions.Independent(torch.distributions.Normal(means, sigmas), 1)
    gmm = torch.distributions.MixtureSameFamily(mix, comp)
    """
    # Circular GMM target distribution
    num_modes = 6
    radius = 10.0
    angles = torch.arange(num_modes, device=device, dtype=torch.float32) * (2*math.pi/num_modes)
    means = torch.stack([radius * angles.cos(), radius * angles.sin()], dim=1)
    weights = torch.full((num_modes,), 1.0/num_modes, device=device)
    sigmas  = torch.full((num_modes, dim), args.sigma_gmm, device=device)

    mix  = torch.distributions.Categorical(weights)
    comp = torch.distributions.Independent(torch.distributions.Normal(means, sigmas), 1)
    gmm  = torch.distributions.MixtureSameFamily(mix, comp)
    """
    # Sample from the GMM
    samples = gmm.sample((args.val_samples,)).cpu()

    # Plot the samples
    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
    #plt.title("Samples from 2D GMM")
    #plt.xlabel("x₁")
    #plt.ylabel("x₂")
    plt.xlim((-1.5,1.5))
    plt.ylim((-1.5,1.5))
    #plt.axis("equal")

    # Save the figure
    plot_path = os.path.join(output_dir, "gmm_samples.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
   
    # Model, optimizer, loss
    fmap      = MLP(dim=dim, out_dim=dim, time_varying=True, w=256).to(device)
    fmap_s    = TorchWrapper(fmap)
    optimizer = optim.Adam(fmap.parameters(), lr=5e-4)
    sink      = SamplesLoss("sinkhorn", p=2, blur=.001)

    # Tracking
    best_val_loss = float('inf')
    val_hist = []
    val_steps = []
    x_gt = gmm.sample((args.val_samples,)).to(device)
    # Training loop
    progress = tqdm(range(args.ntrain), total=args.ntrain)
    for step in progress:
        optimizer.zero_grad()
        # Sample x0 and t
        x0 = gmm.sample((args.batch_size,)).to(device)
        a = args.a
        b = args.b
        D = b - a          # diffusion “width”
        T = args.T        # independent large time‐horizon

        # 1) sample diffusion time t ∈ (0, T)
        t = torch.rand(args.batch_size, 1, device=device) * T
        t = t.clamp(min=args.eps)  # avoid t=0 exactly


        # 2) precompute exponentials using D
        exp_m = torch.exp(-2 * t / D)
        exp_p = torch.exp( 2 * t / D)

        # 3) compute Uniform bounds
        lower = a + (x0 - a) * exp_m
        upper = b - (b - x0) * exp_m

        # 4) sample x_t ∼ Uniform(lower, upper)
        xt = torch.rand_like(x0) * (upper - lower) + lower

        denom = D * (exp_p - 1.0)
        velo_target = (2.0/denom)**(1.0/(args.p - 1.0)) \
              * (xt - x0) \
              * torch.abs(xt - x0)**((2.0 - args.p)/(args.p - 1.0))
        # Model prediction & loss
        out = fmap(torch.cat([xt, t], dim=1))
        loss = F.mse_loss(out, velo_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fmap.parameters(), 1.0)
        optimizer.step()

        # Validation
        if (step + 1) % args.val_interval == 0:
            with torch.no_grad():
                x0_val = gmm.sample((args.val_samples,)).to(device)
                eps_val = torch.randn_like(x0_val)
                xT = torch.rand_like(x0_val) * (args.b - args.a) + args.a










                ode_func = ODEWrapper(fmap_s).to(device)
                t_vals = torch.linspace(T, args.eps, args.num_steps_eval, device=device)
                x_gen = odeint(ode_func, xT, t_vals, method='dopri5')[-1]
                plot_trajectories(x_gen.cpu().numpy(), x_gt.cpu().numpy(), epoch=step+1, out_dir=output_dir)


                nll_val = -gmm.log_prob(x_gen).mean()
                val_loss = sink(x_gen, x_gt)
                print("NLL ", nll_val)
                val_hist.append(val_loss.item())
                val_steps.append(step+1)
                progress.write(f"Step {step+1}: Val Loss = {val_loss:.5f}")
                if nll_val < best_val_loss:
                    best_val_loss = nll_val
                    torch.save(fmap.state_dict(), os.path.join(output_dir, 'best_model.pt'))
                    progress.write(f"→ New best @ step {step+1}")
               
            
   
 
    # Load best model & generate GIF
    fmap.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    visualize_and_save(fmap_s, gmm,T, output_dir,epsilon = args.eps)
