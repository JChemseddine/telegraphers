import torch
import torch.nn as nn
import torch.nn.functional as F

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        # raw weights are unconstrained
        self.weight_raw = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight_raw)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w_pos = F.softplus(self.weight_raw)        # strictly positive
        return F.linear(x, w_pos, self.bias)

class NetScheduler(nn.Module):
    def __init__(self, num_hidden_layers: int):
        super().__init__()
        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, num_hidden_layers)
        self.l3 = PositiveLinear(num_hidden_layers, 1)
        self.activation = torch.nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l1(x) + self.l3(self.activation(self.l2(self.l1(x))))


class normal(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_raw = (-net(t.unsqueeze(1) )).flatten()

        # anchor values at the boundaries         (use same device as t)
        y0 =(-net(torch.tensor([0.0], device=t.device) )).item()
        y1 = (-net(torch.tensor([T],    device=t.device))).item()

        # linear transform s.t.  y_raw==y0 → 1  and  y_raw==y1 → 0
        denom = y0 - y1
        #if abs(denom) < 1e-12:                      # avoid divide-by-zero
        #    raise ValueError("y0 and y1 are (almost) equal; cannot rescale.")

        y = (torch.sigmoid(-(y_raw - y1) / denom  ) - 0.5)  / (torch.sigmoid(torch.tensor([(y0 - y1) / denom], device=t.device) )   -0.5)  + 1       # now y(0)=1, y(1)=0
        return y

if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    net = NetScheduler(num_hidden_layers=1024)
    T = 10.0  # Example time constant
    net.to('mps')  # Move to MPS device if available
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    t = torch.linspace(0, T/2, 100, device='mps').unsqueeze(1)  # Example input tensor
    for epoch in range(10000):
        optimizer.zero_grad()
        output = net(t)
        loss = F.mse_loss(output, torch.ones_like(output))
        loss.backward()
        optimizer.step()
        # dprint(loss)
    

    with torch.no_grad():
        t = torch.linspace(0, T, 100, device='mps')
        
        y_raw = (-net(t.unsqueeze(1) )).flatten()

        # anchor values at the boundaries         (use same device as t)
        y0 =(-net(torch.tensor([0.0], device=t.device) )).item()
        y1 = (-net(torch.tensor([T],    device=t.device))).item()

        # linear transform s.t.  y_raw==y0 → 1  and  y_raw==y1 → 0
        denom = y0 - y1
        #if abs(denom) < 1e-12:                      # avoid divide-by-zero
        #    raise ValueError("y0 and y1 are (almost) equal; cannot rescale.")

        y = (torch.sigmoid(-(y_raw - y1) / denom  ) - 0.5)  / (torch.sigmoid(torch.tensor([(y0 - y1) / denom], device=t.device) )   -0.5)  + 1       # now y(0)=1, y(1)=0
        print(torch.sigmoid(torch.tensor([(y0 - y1) / denom], device=t.device) )   )
        # optional: keep everything inside [0,1]
        #y = y.clamp(0., 1.)
        plt.figure(figsize=(8, 8))

        plt.plot(t.cpu(),y.cpu())
        plt.show()
        g = -net(t.unsqueeze(1)).flatten()                     # raw network output

        # boundary values (detached, so they act as constants)
        g0 = -net(torch.tensor([0.0], device=t.device)).item()
        g1 = -net(torch.tensor([T],    device=t.device)).item()

        # --------------------------------------------------------
        # robust affine mapping:  g==g0  ➜  z=+1 ,  g==g1  ➜  z=0
        # --------------------------------------------------------
        eps = 1e-6                                             # small safety margin
        denom = g0 - g1
        if abs(denom) < eps:                                   # avoid blow-ups
            denom = eps if denom >= 0 else -eps                # keep the sign

        z = (g - g1) / denom                                   # z(0)=+1 , z(T)=0
        y = torch.sigmoid(-z) - 0                                   # squash to (0,1)

        # final numeric guard (not usually necessary, but free):
        y = y.clamp(0.0, 1.0)

        # --------------------------------------------------------
        # visualisation
        # --------------------------------------------------------
        plt.figure(figsize=(8, 8))
        print(f"g0={g0:.4f}, g1={g1:.4f}, denom={denom:+.4e}")
        plt.plot(t.cpu(), y.cpu())
        plt.xlabel("t")
        plt.ylabel("y")
        plt.title("Rescaled output (0,1)")
        plt.show()

        