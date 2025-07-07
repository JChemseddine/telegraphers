import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch import optim
from tqdm import tqdm
from torchdiffeq import odeint
from kac_utils_radial import *
from velo_utils_radial import *
from ode_utils2d import *
from geomloss import SamplesLoss
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from action_model import *

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
parser = argparse.ArgumentParser(description="Train velocity model with validation and GIF creation.")
parser.add_argument('--a', type=float, default=1.0, help="Scale parameter a")
parser.add_argument('--c', type=float, default=10.0, help="Scale parameter c")
parser.add_argument('--T', type=float, default=100.0, help="Time horizon T")
parser.add_argument('--ntrain', type=int, default=100000, help="Number of training steps")
parser.add_argument('--batch_size', type=int, default=512, help="Training batch size")
parser.add_argument('--val_interval', type=int, default=20000, help="Validate every N steps")
parser.add_argument('--val_samples', type=int, default=500, help="Number of samples for validation MMD")
parser.add_argument('--num_steps_eval', type=int, default=200, help="Number of ODE steps for validation")
parser.add_argument('--results_dir', type=str, default="results2d", help="Base directory for results")
parser.add_argument('--eps', type=float, default=0, help="Epsilon Boundary")
parser.add_argument('--seed', type=int, default=0, help="Random seed (default: 0)")
args = parser.parse_args()

args.num_steps_eval = int(args.num_steps_eval * args.T)

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Create result subfolder
subdir = f"results_{args.a}_{args.c}_{args.T}"
args.results_dir= f"results2dRAD_{args.eps}"
output_dir = os.path.join(args.results_dir, subdir)
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# Setup
# --------------------------------------------------
dim = 100

device = 'mps'#"cuda"

a = torch.nn.Parameter(torch.tensor(args.a, device=device))
c = torch.nn.Parameter(torch.tensor(args.c, device=device))
T = args.T
batch_size = args.batch_size    
# GMM target distribution
weights = torch.tensor([0.2]*5, device=device)
means = torch.rand(5, dim, device=device) * 4 - 2
sigmas = torch.ones(5, dim, device=device) * .1
mix = torch.distributions.Categorical(weights)
comp = torch.distributions.Independent(torch.distributions.Normal(means, sigmas), 1)
gmm = torch.distributions.MixtureSameFamily(mix, comp)

# Model and optimizer
fmap = MLP(dim=dim, out_dim=dim, time_varying=True, w=256).to(device)
fmap_s = TorchWrapper(fmap)
optimizer = optim.Adam(fmap.parameters(), lr=5e-4)

# Loss for training and validation (energy distance as proxy for MMD)
sink = SamplesLoss("energy", p=2, blur=1)

# Tracking
best_val_loss = float('inf')
val_hist = []
val_steps = []
def sample_time_inverse_sqrt(batch_size, T, device='cpu'):
    u = torch.rand(batch_size, 1, device=device)
    t = T * (u ** 2)  # CDF of 1/sqrt(t) over [0, T] is ~sqrt(t)/sqrt(T)
    return t

def get_kac_weight(t, a, c, w_min=0.1, w_max=10.0, eps=1e-6):
    """
    Weight ∝ 1 / Var[τ_t] ≈ a / (c^2 t)
    """
    raw_weight = a / (c**2 * t + eps)
    return raw_weight.clamp(min=w_min, max=w_max).detach()
# --------------------------------------------------
# Training loop with validation
# --------------------------------------------------
progress_bar = tqdm(range(args.ntrain), total=args.ntrain)
for step in progress_bar:
    # Zero grad
    optimizer.zero_grad()

    # Sample training batch
    x0 = gmm.sample((args.batch_size,)).to(device)
    # Sample time and trajectory
    t        = torch.rand(batch_size, 1, device=device) * T           # (batch,1)
    t_vec    = t.squeeze(1)                                           # (batch,)
    s_k, sums = create_sk(a, t_vec, T)
   
    tau      = sample_trj(t_vec, s_k, sums, a, T)                     # (batch,)

    # Random direction on unit sphere
    u        = torch.randn(batch_size, dim, device=device)
    u       /= u.norm(dim=1, keepdim=True)                            # normalize

    # Final forward sample
    xt       = x0 + (c * tau).unsqueeze(1) * u   

    # Compute model output and velocity loss
    out_t = fmap(torch.cat([xt, t], dim=1))
    velo = compute_velocity_nd(
    xt - x0,            # ⟂ displacement in ℝ^d
    t,                  # (batch_size,1)
    a.detach(),         # scalar
    c.detach(),         # scalar
    epsilon=args.eps
    )

    w = get_kac_weight(t, a, c)
    loss = (F.mse_loss(out_t, velo, reduction='none')).mean() #/ w.mean()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(fmap.parameters(), 1.0)
    optimizer.step()

    # Validation
    if (step + 1) % args.val_interval == 0:
        with torch.no_grad():
            # Sample from model via simple Euler or ODE integration
            num_val = args.val_samples
            # Initialize at t = T from GMM
            t_end    = torch.full((num_val, 1), T, device=device)            # (num_val,1)
            t_vec_T  = t_end.squeeze(1)                                      # (num_val,)
            s_k, sums = create_sk(a, t_vec_T, T)                             # (num_val,K)
            tau_T    = sample_trj(t_vec_T, s_k, sums, a, T)                  # (num_val,)

            u        = torch.randn(num_val, dim, device=device)
            u       /= u.norm(dim=1, keepdim=True)

            x_T      = 0#gmm.sample((num_val,)).to(device)                     # (num_val, dim)
            x_T     += (c * tau_T).unsqueeze(1) * u                          # (num_val, dim)
            # Reverse-time integration (placeholder: identity mapping)
            ode_func = ODEWrapper(fmap_s).to(device)
            t_vals = torch.linspace(T, 0, args.num_steps_eval, device=device)
            x_gen = odeint(ode_func, x_T, t_vals, method='euler')[-1]

           
            x_gt = gmm.sample((num_val,)).to(device)

            val_loss = sink(x_gen, x_gt)
            xg = x_gen.cpu().numpy()
            plt.figure(figsize=(4,4))
            plt.scatter(xg[:, 0], xg[:, 1], alpha=0.5, s=5)
            plt.title(f"Generated samples at step {step+1}")
            plt.xlabel("x₁")
            plt.ylabel("x₂")
            fname = os.path.join(output_dir, f"x_gen_step{step+1}.png")
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close()
        val_hist.append(val_loss.item())
        val_steps.append(step + 1)
        progress_bar.write(f"Step {step+1}: Val Loss = {val_loss.item():.5f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(fmap.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            progress_bar.write(f"New best model saved at step {step+1} with val loss {val_loss.item():.5f}")

# --------------------------------------------------
# Plot validation curve
# --------------------------------------------------
plt.figure()
plt.plot(val_steps, val_hist)
plt.xlabel('Training Step')
plt.ylabel('Validation Loss (Energy)')
plt.title('Validation Curve')
plt.savefig(os.path.join(output_dir, 'val_curve.png'))
plt.close()

val_loss_path = os.path.join(output_dir, 'val_loss.txt')
with open(val_loss_path, 'w') as f:
    # write header
    f.write("step\tloss\n")
    for step, loss_val in zip(val_steps, val_hist):
        f.write(f"{step}\t{loss_val:.6f}\n")
    # also record the best overall
    f.write(f"best_val_loss\t{best_val_loss:.6f}\n")

print(f"Validation loss history saved to {val_loss_path}")
# --------------------------------------------------
# Final evaluation and GIF creation using best model
# --------------------------------------------------
# Reload best model
fmap.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))

def visualize_and_save(fmap_s, T, output_dir, num_steps=50, num_samples=5000, dim=2, device='cuda'):
    import os
    from matplotlib import animation
    import numpy as np
    with torch.no_grad():
        # Prepare initial time and samples
        t_vec_T  = t_end.squeeze(1)                                 # (num_samples,)
        s_k, sums = create_sk(a, t_vec_T, T)
        tau_T    = sample_trj(t_vec_T, s_k, sums, a, T)             # (num_samples,)

        u        = torch.randn(num_samples, dim, device=device)
        u       /= u.norm(dim=1, keepdim=True)

        x_T      = 0#gmm.sample((num_samples,)).to(device)
        x_T     += (c * tau_T).unsqueeze(1) * u
        ode_func = ODEWrapper(fmap_s).to(device)

    # Nonlinear time stepping: slow down near t=0
    slow_fraction = 0.2  # Slow down in last 20%
    fast_steps = int(num_steps * (1 - slow_fraction))
    slow_steps = num_steps - fast_steps

    # Time values: nonlinear mapping (quadratic)
    t_fast = np.linspace(T, T * slow_fraction, fast_steps, endpoint=False)
    t_slow = np.linspace(T * slow_fraction, 0.0, slow_steps)
    t_vals_np = np.concatenate([t_fast, t_slow])
    t_vals = torch.tensor(t_vals_np, device=device, dtype=torch.float32)

    with torch.no_grad():
        x_traj = odeint(ode_func, x_T, t_vals, method='dopri5')  # shape: (num_steps, num_samples, dim)

    # EMA state for dynamic axis limits
    smoothed_xlim = None
    smoothed_ylim = None
    alpha = 0.09  # smoothing factor

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        nonlocal smoothed_xlim, smoothed_ylim
        ax.clear()
        x_frame = x_traj[frame].detach().cpu().numpy()

        # Compute padding for view limits
        min_vals = x_frame.min(axis=0)
        max_vals = x_frame.max(axis=0)
        padding = 0.2 * (max_vals - min_vals + 1e-3)
        xlim_raw = (min_vals[0] - padding[0], max_vals[0] + padding[0])
        ylim_raw = (min_vals[1] - padding[1], max_vals[1] + padding[1])

        # Smooth limits
        if smoothed_xlim is None:
            smoothed_xlim = xlim_raw
            smoothed_ylim = ylim_raw
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
        ax.scatter(x_frame[:, 0], x_frame[:, 1], alpha=0.5, c='blue', s=2)

    # Pause on final frame
    pause_frames = 20
    frame_indices = list(range(num_steps)) + [num_steps - 1] * pause_frames

    ani = animation.FuncAnimation(fig, update, frames=frame_indices, repeat=False)
    gif_path = os.path.join(output_dir, 'trajectory.gif')
    ani.save(gif_path, writer='imagemagick', fps=5)
    plt.close(fig)

    #return gif_path


visualize_and_save(fmap_s, T, output_dir)
