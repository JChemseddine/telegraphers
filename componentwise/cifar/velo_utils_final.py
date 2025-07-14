import torch
from torch.special import i0e, i1e

def _ratio_I1_I0(z: torch.Tensor) -> torch.Tensor:
    """
    Compute I1(z)/I0(z) robustly for all z >= 0, using
    the series expansion for small z and the scaled Bessels elsewhere.
    """
    small = z < 1e-12#1e-6
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
    """
    Compute v(x,t) = J/u for the 1D Kac process with
    • series‐expanded Bessel ratio for z small
    • a hard ±c boundary layer of thickness eps·t.

    Now `c` may be a scalar or a tensor of shape broadcastable with x and t.
    """
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

def compute_velocity_float64(x: torch.Tensor,
                     t: torch.Tensor,
                     a: float,
                     c: float) -> torch.Tensor:
    """
    Compute v(x, t) = J(t,x)/u(t,x) for the 1D Kac process,
    vectorized in both x and t.

    x: Tensor of shape (...), e.g. (batch, N)
    t: Tensor broadcastable to x, e.g. (batch, 1) or same shape as x
    a, c: scalar floats

    Returns v of same shape as x.
    """
    # ensure float64 and same device
    x = x.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64, device=x.device)

    # make a, c scalars
    a = float(a)
    c = float(c)

    # compute r = sqrt(clamped(c^2 t^2 - x^2))
    r2 = c * c * t * t - x * x
    r = torch.sqrt(torch.clamp(r2, min=0.0))

    # scaled argument for Bessel ratio
    z = (a / c) * r
    R = _ratio_I1_I0(z)      # I1/I0
    invR = 1.0 / R           # I0/I1

    # continuous part
    denom = t + (r / c) * invR
    v_cont = x / denom

    # atomic boundary (|x| >= c t) → v = ±c
    mask = x.abs() >= (c * t)
    v = torch.where(mask, torch.sign(x) * c, v_cont)

    return v
