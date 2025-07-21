import torch
from torch.special import i0e, i1e

def _ratio_I1_I0(z: torch.Tensor) -> torch.Tensor:

    small = z < 1e-12
    # cubic Maclaurin: I1/I0 ≈ z/2 - z^3/16
    r_series = 0.5 * z #- (z**3) * (1.0/16.0)
    r_full   = i1e(z) / i0e(z)
    return torch.where(small, r_series, r_full)

def compute_velocity(x: torch.Tensor,
                     t: torch.Tensor,
                     a: float,
                     c: torch.Tensor,
                     epsilon: float = 1e-6,
                     T: float = 1.0) -> torch.Tensor:

    # cast inputs to float64 for stability
    x = x.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64, device=x.device)
    eps = float(epsilon)
    a = float(a)

    # ensure c is a float64 tensor on the correct device
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=torch.float64, device=x.device)
    else:
        c = c.to(dtype=torch.float64, device=x.device)

    # radial part: r = sqrt((c t)^2 - x^2)
    r2 = (c * t)**2 - x**2
    r  = torch.sqrt(torch.clamp(r2, min=0.0))

    # argument for Bessel ratio
    z = (a / c) * r
    R = _ratio_I1_I0(z)

    # denom = t + (r/c)*(1/R), except for very small z
    tiny = z < 1e-12#1e-6
    invR = 1.0 / R
    denom_general = t + (r / c) * invR
    denom_tiny   = t + 2.0 / a
    denom = torch.where(tiny, denom_tiny, denom_general)

    # continuous velocity
    v_cont = x / denom

    # enforce |v| ≤ c in an eps-thick layer at the light-cone
    # threshold = c*t - eps*(t/T)*c
    threshold = c * t - eps * (t / T) * c
    boundary  = torch.sign(x) * c
    mask      = x.abs() >= threshold

    v = torch.where(mask, boundary, v_cont)
    return v.to(dtype=torch.float32)
