import numpy as np
import torch
from torch import Tensor
from scipy.special import i0e, i1e

class TorchKacConstantSampler:
    """
    O(1) mixture sampler for the 1D Kac displacement.
    Includes the atomic component at ±c*t (probability e^{-a t})
    plus the continuous part via a precomputed inverse-CDF lookup.
    """
    def __init__(
        self,
        a: float,
        c: float,
        T: float,
        M: int,
        K: int = 1024,
        device=None,
        dtype=torch.float32,
    ):
        self.a = a
        self.c = c
        self.beta = a / c
        self.T = T
        self.M = M
        self.K = K
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        t_grid = np.linspace(0.0, T, M + 1)
        U = np.linspace(0.0, 1.0, K + 1)
        X_table = np.zeros((M + 1, K + 1), dtype=float)
        F_table = np.zeros((M + 1, K + 1), dtype=float)
        eps_ct = 1e-16

        for j, t in enumerate(t_grid):
            ct = c * t
            if ct < eps_ct:
                X_table[j] = 0.0
                F_table[j] = U
                continue

            norm = -np.expm1(-a * t)
            r = np.sqrt(np.maximum(ct * ct - (ct * U) ** 2, 0.0))
            z = self.beta * r
            exp_fac = np.exp(z - a * t)

            term1 = self.beta * exp_fac * i0e(z)
            small = (z <= 1e-6)
            ratio = np.empty_like(z)
            ratio[~small] = i1e(z[~small]) / r[~small]
            ratio[small] = self.beta * (0.5 + (z[small]**2)/16.0 + (z[small]**4)/384.0) * np.exp(-z[small])
            term2 = self.beta * ct * exp_fac * ratio
            Kz = 0.5 * (term1 + term2)
            f = 2.0 * (Kz / norm) * ct

            dU = U[1:] - U[:-1]
            F = np.empty(K + 1, dtype=float)
            F[0] = 0.0
            F[1:] = np.cumsum(0.5 * (f[:-1] + f[1:]) * dU)
            if F[-1] > 0:
                F /= F[-1]
            else:
                F = U

            X_table[j] = ct * U
            F_table[j] = F

        quantiles = np.linspace(0.0, 1.0, K + 1)
        invC = np.empty_like(X_table)
        for j in range(M + 1):
            invC[j] = np.interp(quantiles, F_table[j], X_table[j])

        self.t_grid = torch.tensor(t_grid, device=self.device, dtype=self.dtype)
        self.invC_table = torch.tensor(invC, device=self.device, dtype=self.dtype)

    def sample(self, t: Tensor, dim: int = 1) -> Tensor:
        orig_shape = t.shape
        t_flat = t.reshape(-1).to(self.device).to(self.dtype)
        N = t_flat.shape[0]
        B = N * dim

        t_exp = t_flat.unsqueeze(1).expand(-1, dim).reshape(-1)
        mix_u = torch.rand(B, device=self.device, dtype=self.dtype)
        cont_u = torch.rand(B, device=self.device, dtype=self.dtype)
        p0 = torch.exp(-self.a * t_exp)
        is_atomic = mix_u < p0

        dt = self.T / self.M
        j = torch.clamp((t_exp / dt).floor().long(), 0, self.M - 1)
        alpha = (t_exp - self.t_grid[j]) / dt

        ut = torch.clamp(cont_u, max=(self.K - 1) / self.K) * self.K
        k = ut.floor().long()
        frac = ut - k

        x0 = self.invC_table[j,   k]
        x1 = self.invC_table[j,   k + 1]
        y0 = self.invC_table[j + 1, k]
        y1 = self.invC_table[j + 1, k + 1]
        xj  = x0 + frac * (x1 - x0)
        xj1 = y0 + frac * (y1 - y0)
        x_cont = xj + alpha * (xj1 - xj)

        mag = torch.where(is_atomic, self.c * t_exp, x_cont)
        signs = torch.where(torch.rand(B, device=self.device) < 0.5, 1.0, -1.0).to(self.dtype)
        return (signs * mag).view(*orig_shape, dim)
# Usage note:
# sampler = TorchKacConstantSampler(a=1., c=1., T=100., M=10, K=1024)
# t = torch.rand(32)*sampler.T
# x = sampler.sample(t, dim=3)
# print(x.shape)  # torch.Size([32,3])
