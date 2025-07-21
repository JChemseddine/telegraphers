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
            torch.nn.ReLU(),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


