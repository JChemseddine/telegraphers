import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1]
class MLP2(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class MLP3(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
           
        )

    def forward(self, x):
        return self.net(x)

class MLP4(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
           
        )

    def forward(self, x):
        return self.net(x)
class MLP4Residual(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=True):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        
        # Shared layers for x and t
        self.net_pre = torch.nn.Sequential(
            torch.nn.Linear(dim, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU()
        )
        
        # Residual pathway for time component t
        self.time_net = torch.nn.Sequential(
            torch.nn.Linear(1, w),  # Processing time separately
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU()
        )
        
        # Post-residual connection layers
        self.net_post = torch.nn.Sequential(
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim)
        )

    def forward(self, x,t):
        # Separate the time component (last dimension is t)
        #t = x[:, -1].unsqueeze(1)  # Extract the 1D time t
        #x_only = x[:, :-1]          # Extract x (d-dimensional)
        
        # Pass the concatenated x and t through the first part of the network
        x_emb = self.net_pre(x)  # Includes both x and t

        # Process t in the residual pathway
        t_emb = self.time_net(t)

        # Add the residual time embedding back to the x embedding
        combined_emb = x_emb + t_emb  # Residual connection for t
        
        # Pass through the final layers
        output = self.net_post(combined_emb)
        return output
class MLPB(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            nn.BatchNorm1d(w),

            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            nn.BatchNorm1d(w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            nn.BatchNorm1d(w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            nn.BatchNorm1d(w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
           
        )

    def forward(self, x):
        return self.net(x)
class FourierFeatures(nn.Module):
    def __init__(self, in_dim, n_features=256, scale=10.0):
        super().__init__()
        self.B = torch.randn(in_dim, n_features,device="cuda") * scale

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierMLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, n_fourier_features=256, scale=10.0, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        
        # Fourier feature mappings for x and time (if time_varying)
        self.fourier_x = FourierFeatures(dim, n_features=n_fourier_features, scale=scale)
        
        if time_varying:
            self.fourier_t = FourierFeatures(1, n_features=n_fourier_features, scale=scale)
            self.total_fourier_features = n_fourier_features * 4  # 2 for x (sin, cos) and 2 for t (sin, cos)
        else:
            self.total_fourier_features = n_fourier_features * 2  # 2 for x (sin, cos)

        self.net = nn.Sequential(
            nn.Linear(self.total_fourier_features, w),
            nn.SiLU(),
            nn.Linear(w, w),
            nn.SiLU(),
            nn.Linear(w, w),
            nn.SiLU(),
            nn.Linear(w, w),
            nn.SiLU(),
            nn.Linear(w, out_dim),
        )

    def forward(self, x, t=None):
        x = self.fourier_x(x)  # Fourier transform for x
        
        if self.time_varying:
            t = self.fourier_t(t)  # Fourier transform for time
            x = torch.cat([x, t], dim=-1)  # Concatenate x and time embeddings
        
        return self.net(x)
