import torch
import numpy as np
from scipy.special import i0 as np_i0, i1 as np_i1
from scipy.integrate import quad
from scipy.optimize import bisect
from torch.special import i0e as torch_i0e, i1e as torch_i1e


def build_half_ziggurat_numpy(p_u, n, tol=1e-8):
    """
    Build half-Ziggurat tables in numpy for a PDF p_u(u) supported on [0,1].
    Returns arrays u[0..n] (descending from 1 to 0) and y[0..n]=p_u(u[i]).
    """
    u = np.zeros(n+1)
    y = np.zeros(n+1)
    u[0], y[0] = 1.0, p_u(1.0)
    u[n], y[n] = 0.0, p_u(0.0)
    A = 1.0 / n
    def tail_mass(u0):
        res, _ = quad(p_u, u0, 1.0)
        return res
    for i in range(1, n):
        target = (n - i) * A
        u_i = bisect(lambda uu: tail_mass(uu) - target, 0.0, 1.0, xtol=tol)
        u[i] = u_i
        y[i] = p_u(u_i)
    return u, y

class TorchKacZigguratSampler:
    def __init__(self, a=1.0, c=1.0, T=100.0, M=10000, n=128, device=None, dtype=torch.float32):
        """
        Build Ziggurat tables for the continuous part of 1D Kac density
        on a uniform time-grid 0..T with M intervals (M+1 points) and n strips.
        """
        self.a, self.c = a, c
        self.beta = a / c
        self.T, self.M, self.n = T, M, n
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Uniform time grid
        self.t_grid = np.linspace(0, T, M+1)
        self.dt = T / M

        # Build tables in numpy
        U_np = np.zeros((M+1, n+1), dtype=float)
        Y_np = np.zeros((M+1, n+1), dtype=float)
        for j in range(1, M+1):
            t_j = self.t_grid[j]
            ct = c * t_j
            exp_at = np.exp(-a * t_j)
            norm = 1.0 - exp_at
            def p_u(u):
                x = ct * u
                r = np.sqrt(max(ct*ct - x*x, 0.0))
                K = 0.5 * exp_at * (
                    self.beta * np_i0(self.beta * r)
                    + self.beta * ct * (np_i1(self.beta * r) / (r if r>0 else 1.0))
                )
                return 2.0 * (K / norm) * ct
            Uj, Yj = build_half_ziggurat_numpy(p_u, n)
            U_np[j], Y_np[j] = Uj, Yj

        # Move tables to torch
        self.U = torch.from_numpy(U_np).to(device=self.device, dtype=self.dtype)
        self.Y = torch.from_numpy(Y_np).to(device=self.device, dtype=self.dtype)
        self.t_grid_torch = torch.from_numpy(self.t_grid).to(device=self.device, dtype=self.dtype)

    def sample_continuous(self, t):
        """
        Sample the continuous part of the 1D Kac displacement at times t (batch,).
        """
        t = t.to(device=self.device, dtype=self.dtype)
        batch = t.shape[0]

        # Interpolate Ziggurat tables for this t
        j = torch.clamp((t / self.dt).floor().long(), 0, self.M - 1)
        alpha = ((t - self.t_grid_torch[j]) / self.dt).unsqueeze(1)
        Uj = (1 - alpha) * self.U[j] + alpha * self.U[j+1]
        Yj = (1 - alpha) * self.Y[j] + alpha * self.Y[j+1]

        out = torch.empty(batch, device=self.device, dtype=self.dtype)
        mask = torch.ones(batch, dtype=torch.bool, device=self.device)

        while mask.any():
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            # Propose within strips
            i = torch.randint(0, self.n, (idx.shape[0],), device=self.device)
            u_low = Uj[idx, i+1]
            u_high = Uj[idx, i]
            u = u_low + (u_high - u_low) * torch.rand_like(u_low)
            v = Yj[idx, i] * torch.rand_like(u)

            # Compute true half-density p_u_true at u
            t_sel = t[idx]
            ct = self.c * t_sel
            x = ct * u
            r2 = ct * ct - x * x
            r = torch.sqrt(torch.clamp(r2, min=0.0))
            z = self.beta * r

            # Use scaled Bessels for stability: exp_factor = exp(-a t) * exp(z)
            exp_factor = torch.exp(z - self.a * t_sel)
            term1 = self.beta * exp_factor * torch_i0e(z)

            # Term2 = beta^2 * ct * exp_factor * (I1(z)/z), avoid dividing by tiny z
            z_tol = 1e-2
            ratio = torch.where(
                z > z_tol,
                torch_i1e(z) / z,
                0.5 + z*z/16.0 + z.pow(4)/384.0
            )
            term2 = (self.beta * self.beta) * ct * exp_factor * ratio

            K = 0.5 * (term1 + term2)
            norm = 1.0 - torch.exp(-self.a * t_sel)
            p_u_true = 2.0 * (K / norm) * ct

            accept = v <= p_u_true
            signs = torch.where(torch.rand_like(u) < 0.5, 1.0, -1.0)
            out[idx[accept]] = signs[accept] * x[accept]

            # Update mask for remaining draws
            new_mask = torch.zeros_like(mask)
            new_mask[idx[~accept]] = True
            mask = new_mask

        return out

    def sample_kac(self, t, dim=1):
        """
        Sample from the full 1D Kac displacement at times t (batch,). 
        Returns shape [batch] if dim=1, else [batch,dim] with independent components.
        """
        batch = t.shape[0]
        # Expand times for multi-dim sampling
        if dim > 1:
            t_exp = t.unsqueeze(1).expand(-1, dim).reshape(-1)
        else:
            t_exp = t

        # Continuous draws
        cont = self.sample_continuous(t_exp)

        # Atom draws via log-uniform compare: log(u) < -a*t avoids underflow
        log_u = torch.log(torch.rand_like(t_exp))
        atom_mask = log_u < -self.a * t_exp
        signs = torch.where(torch.rand_like(t_exp) < 0.5, 1.0, -1.0)
        atom_vals = signs * (self.c * t_exp)

        full = torch.where(atom_mask, atom_vals, cont)

        if dim > 1:
            full = full.view(batch, dim)
        return full
