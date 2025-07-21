#!/usr/bin/env python

import os
import argparse
import random
import itertools

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
from torchdiffeq import odeint

from utils.sample_kac import TorchKacConstantSampler
from utils.velo_utils import *
from utils.ode_utils2d import *
from utils.mlp import MLP
from geomloss import SamplesLoss

import matplotlib.patches as patches
import matplotlib.lines as mlines

def get_args():
    parser = argparse.ArgumentParser(
        description="Train velocity model with validation and GIF creation."
    )
    parser.add_argument('--a', type=float, default=9.0)
    parser.add_argument('--c', type=float, default=3.0)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--sigma_gmm', type=float, default=0.0001)
    parser.add_argument('--ntrain', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--val_interval', type=int, default=20000)
    parser.add_argument('--val_samples', type=int, default=5000)
    parser.add_argument('--num_steps_eval', type=int, default=200)
    parser.add_argument('--results_dir', type=str, default="results2d")
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()


def visualize_and_save(fmap_s, sampler, gmm, T, output_dir,
                       num_steps=50, num_samples=5000, dim=2, device='cuda'):
    slow_frac = 0.2
    fast_steps = int(num_steps * (1 - slow_frac))
    slow_steps = num_steps - fast_steps
    t_fast = np.linspace(T, T * slow_frac, fast_steps, endpoint=False)
    t_slow = np.linspace(T * slow_frac, 0.0, slow_steps)
    t_vals = torch.tensor(
        np.concatenate([t_fast, t_slow]),
        device=device, dtype=torch.float32
    )

    with torch.no_grad():
        t_end = torch.ones(num_samples, 1, device=device) * T
        tau_T = sampler.sample(t_end.squeeze(1), dim=dim).to(device)
        x_T = tau_T + gmm.sample((num_samples,)).to(device)
        ode_func = ODEWrapper(fmap_s).to(device)
        x_traj = odeint(ode_func, x_T, t_vals, method='dopri5')

    fig, ax = plt.subplots(figsize=(6, 6))
    sm_xlim = sm_ylim = None
    alpha = 0.09

    def update(frame):
        nonlocal sm_xlim, sm_ylim
        ax.clear()
        x_frame = x_traj[frame].cpu().numpy()
        min_v = x_frame.min(axis=0)
        max_v = x_frame.max(axis=0)
        pad = 0.2 * (max_v - min_v + 1e-3)
        raw_xlim = (min_v[0] - pad[0], max_v[0] + pad[0])
        raw_ylim = (min_v[1] - pad[1], max_v[1] + pad[1])
        if sm_xlim is None:
            sm_xlim, sm_ylim = raw_xlim, raw_ylim
        else:
            sm_xlim = (
                alpha * raw_xlim[0] + (1 - alpha) * sm_xlim[0],
                alpha * raw_xlim[1] + (1 - alpha) * sm_xlim[1]
            )
            sm_ylim = (
                alpha * raw_ylim[0] + (1 - alpha) * sm_ylim[0],
                alpha * raw_ylim[1] + (1 - alpha) * sm_ylim[1]
            )
        ax.set_xlim(*sm_xlim)
        ax.set_ylim(*sm_ylim)
        ax.set_title(f"t = {t_vals[frame]:.2f}")
        ax.scatter(x_frame[:, 0], x_frame[:, 1], s=2, alpha=0.5, c='blue')

    from matplotlib import animation
    pause = 20
    frames = list(range(num_steps)) + [num_steps - 1] * pause
    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
    gif_path = os.path.join(output_dir, 'trajectory.gif')
    ani.save(gif_path, writer='imagemagick', fps=5)
    plt.close(fig)


def main():
    args = get_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = os.path.join(
        args.results_dir,
        f"results_{args.a}_{args.c}_{args.T}"
    )
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 2

    sampler = TorchKacConstantSampler(
        a=args.a, c=args.c, T=args.T,
        M=10000, K=4096,
        device=device
    )

    grid_pts = list(itertools.product([-1.0, 0.0, 1.0], repeat=dim))
    means = torch.tensor(grid_pts, device=device)
    weights = torch.full((len(means),), 1.0 / len(means), device=device)
    sigmas = torch.ones_like(means) * args.sigma_gmm

    mix = torch.distributions.Categorical(weights)
    comp = torch.distributions.Independent(
        torch.distributions.Normal(means, sigmas), 1
    )
    gmm = torch.distributions.MixtureSameFamily(mix, comp)

    samples = gmm.sample((args.val_samples,)).cpu()
    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.savefig(os.path.join(output_dir, 'gmm_samples.png'),
                bbox_inches='tight', dpi=150)
    plt.close()

    model = MLP(dim=dim, out_dim=dim, time_varying=True, w=256).to(device)
    wrapper = TorchWrapper(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.001)

    best_loss = float('inf')
    val_hist, val_steps = [], []
    x_gt = gmm.sample((args.val_samples,)).to(device)

    for step in tqdm(range(args.ntrain), desc="Training"):
        optimizer.zero_grad()
        x0 = gmm.sample((args.batch_size,)).to(device)
        t = torch.rand(args.batch_size, 1, device=device) * args.T
        tau = sampler.sample(t.squeeze(1), dim=dim).to(device)
        xt = x0 + tau

        out = model(torch.cat([xt, t], dim=1))
        velo = compute_velocity(xt - x0, t, args.a, args.c,
                                epsilon=args.eps, T=args.T)
        loss = F.mse_loss(out, velo).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % args.val_interval == 0:
            with torch.no_grad():
                t_end = torch.ones(args.val_samples, 1,
                                   device=device) * args.T
                tau_T = sampler.sample(t_end.squeeze(1), dim=dim).to(device)
                x_T = tau_T + gmm.sample((args.val_samples,)).to(device)
                ode_func = ODEWrapper(wrapper).to(device)
                t_vals = torch.linspace(
                    args.T, 0, args.num_steps_eval, device=device
                )
                x_gen = odeint(ode_func, x_T, t_vals, method='dopri5')[-1]

                nll = -gmm.log_prob(x_gen).mean()
                val_loss = nll

                val_hist.append(val_loss.item())
                val_steps.append(step + 1)

                if nll < best_loss:
                    best_loss = nll
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, 'best_model.pt'))
            xg = x_gen.cpu().numpy()
            xgt = x_gt.cpu().numpy()

            zoom_range = 0.1

            fig, (ax_main, ax_zoom) = plt.subplots(
                1, 2,
                figsize=(8, 4),
                gridspec_kw={'width_ratios': [3.2, 1], 'wspace': 0.01},
                subplot_kw={'aspect': 'equal'}
            )

            # --- main plot ---
            ax_main.scatter(xg[:,0], xg[:,1], s=2, alpha=0.5, label='gen')
            ax_main.scatter(xgt[:,0], xgt[:,1], s=2, alpha=0.5, c='r', label='gt')
            ax_main.set_xlim(-1.5, 1.5)
            ax_main.set_ylim(-1.5, 1.5)
            #ax_main.set_xlabel("x₁")
            #ax_main.set_ylabel("x₂")
            #ax_main.legend(loc='upper left')

            # draw the dashed square
            rect = patches.Rectangle(
                (-zoom_range, -zoom_range),
                2*zoom_range, 2*zoom_range,
                linewidth=1, edgecolor='gray',
                facecolor='none', linestyle='--'
            )
            ax_main.add_patch(rect)

            # --- zoom plot ---
            ax_zoom.scatter(xg[:,0], xg[:,1], s=2, alpha=0.5)
            ax_zoom.scatter(xgt[:,0], xgt[:,1], s=2, alpha=0.5, c='r')
            ax_zoom.set_xlim(-zoom_range, zoom_range)
            ax_zoom.set_ylim(-zoom_range, zoom_range)
            ax_zoom.set_xticks([])
            ax_zoom.set_yticks([])
            ax_zoom.set_title(f"Zoomed (±{zoom_range})")
            #ax_zoom.set_xlabel("x₁")
            #ax_zoom.set_ylabel("x₂")

            # --- corner-to-corner lines ---
            # helper: data→figure coords via main axes
            df_to_fig = lambda x, y: fig.transFigure.inverted().transform(
                ax_main.transData.transform((x,y))
            )

            # compute square corners in figure space
            fig_UR = df_to_fig( zoom_range,  zoom_range)  # upper-right
            fig_LR = df_to_fig( zoom_range, -zoom_range)  # lower-right

            # get zoom axes bbox (figure space)
            bb = ax_zoom.get_position()
            fig_zoom_UL = (bb.x0, bb.y1)
            fig_zoom_LL = (bb.x0, bb.y0)

            # draw dotted lines
            line1 = mlines.Line2D(
                [fig_UR[0], fig_zoom_UL[0]],
                [fig_UR[1], fig_zoom_UL[1]],
                linestyle=':', color='gray', transform=fig.transFigure
            )
            line2 = mlines.Line2D(
                [fig_LR[0], fig_zoom_LL[0]],
                [fig_LR[1], fig_zoom_LL[1]],
                linestyle=':', color='gray', transform=fig.transFigure
            )
            fig.add_artist(line1)
            fig.add_artist(line2)
            fname = os.path.join(output_dir, f"zoomed_{step}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
    plt.figure()
    plt.plot(val_steps, val_hist)
    plt.xlabel('Step')
    plt.ylabel('Validation Loss')
    plt.savefig(os.path.join(output_dir, 'val_curve.png'))
    plt.close()

    with open(os.path.join(output_dir, 'val_loss.txt'), 'w') as f:
        f.write("step\tloss\n")
        for s, l in zip(val_steps, val_hist):
            f.write(f"{s}\t{l:.6f}\n")
        f.write(f"best_val_loss\t{best_loss:.6f}\n")

    best_state = os.path.join(output_dir, 'best_model.pt')
    model.load_state_dict(torch.load(best_state))

    visualize_and_save(wrapper, sampler, gmm, args.T, output_dir,
                       num_steps=50, num_samples=args.val_samples,
                       dim=dim, device=device)


if __name__ == '__main__':
    main()
