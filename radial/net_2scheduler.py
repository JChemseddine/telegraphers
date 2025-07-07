import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotoneNonlinear01(nn.Module):
    """
    Smooth, strictly monotone–decreasing NN on t∈[0,1]
    with exact boundary values y(0)=1, y(1)=0.
    """

    def __init__(self, hidden_sizes=(32, 32, 32)):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h, bias=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1, bias=True))
        self.layers = nn.ModuleList(layers)

        #  optional: small-variance init so F(0) ≈ 0, F(1) ≈ 1 before training
        for lin in self.layers:
            nn.init.normal_(lin.weight, 0.0, 0.1)
            nn.init.normal_(lin.bias,   0.0, 0.1)

    # ---- helper ----------------------------------------------------------
    @staticmethod
    def _positive(x):
        """Soft-plus keeps weights & biases ≥0."""
        return F.softplus(x)

    def _F(self, t):
        """
        Monotone-increasing sub-network: t ↦ F(t)   (F′(t) ≥ 0).
        """
        x = t
        for i, lin in enumerate(self.layers):
            w = self._positive(lin.weight)
            b = self._positive(lin.bias)
            x = F.linear(x, w, b)
            if i < len(self.layers) - 1:        # hidden layers
                x = F.softplus(x)
        return x.squeeze(-1)                    # shape (batch,)

    # ---- forward ---------------------------------------------------------
    def forward(self, t):
        """
        y(t) = 1 - (F(t) - F(0)) / (F(1) - F(0))
        guarantees y(0)=1, y(1)=0 and monotone ↓.
        """
        t = t.unsqueeze(-1)                     # ensure shape (batch,1)
        F_t = self._F(t)
        F_0 = self._F(t.new_zeros(1))           # scalar
        F_1 = self._F(t.new_ones(1))            # scalar

        denom = (F_1 - F_0).clamp_min(1e-7)     # avoid divide-by-0
        y = 1.0 - (F_t - F_0) / denom
        return y
if __name__ == "__main__":
    net = MonotoneNonlinear01()
    t = torch.linspace(0, 1, 11)
    y = net(t)
    
    import matplotlib.pyplot as plt
    plt.plot(t.numpy(), y.detach().numpy(), marker='o')
    plt.show()
    
    
device      = "mps"#"cuda" if torch.cuda.is_available() else "cpu"
batch_size  = 64
net         = nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                            nn.Linear(32, 32), nn.ReLU(),
                            nn.Linear(32, 1)).to(device)
optim       = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(3_000):
    idx     = torch.rand(batch_size, 1, device=device)   # fresh batch
    preds   = net(idx)
    target  = torch.ones_like(preds)                     # shape & dtype match
    loss    = F.mse_loss(preds, target)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if epoch % 500 == 0:
        print(f"epoch {epoch:5d}   loss {loss.item():.6f}   "
              f"mean(pred) {preds.mean().item():.3f}")
y = net(t.to('mps')).cpu()
    
import matplotlib.pyplot as plt
plt.plot(t.numpy(), y.detach().numpy(), marker='o')
plt.show()