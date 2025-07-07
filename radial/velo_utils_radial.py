import torch
from torch.special import i0e, i1e

def _ratio_I1_I0(z: torch.Tensor) -> torch.Tensor:
    """
    Compute I1(z)/I0(z) robustly for all z >= 0, using
    the series expansion for small z and the scaled Bessels elsewhere.
    """
    small = z < 0.01
    # cubic Maclaurin: I1/I0 ≈ z/2 - z^3/16
    r_series = 0.5 * z - (z**3) * (1.0/16.0)
    r_full   = i1e(z) / i0e(z)
    return torch.where(small, r_series, r_full)

def compute_velocity_nd(
    x: torch.Tensor,
    t: torch.Tensor,
    a: float,
    c: float,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Radial Kac velocity in ℝ^d:

      x : (..., d)
      t : (..., 1) or broadcastable to x’s leading dims
      a, c, eps : scalars

    Returns v of same shape as x.
    """
    # -- cast & unpack --
    x = x.to(dtype=torch.float64)
    t = t.to(dtype=torch.float64, device=x.device)
    a = float(a); c = float(c); eps = float(epsilon)

    # -- radial distance inside light‐cone --
    #    R^2 = c^2 t^2 - ||x||^2
    norm2 = (x**2).sum(dim=-1, keepdim=True)
    r2    = c*c * t*t - norm2
    r     = torch.sqrt(torch.clamp(r2, min=0.0))

    # -- Bessel ratio & denom --
    z       = (a/c) * r
    R_ratio = _ratio_I1_I0(z)                 # I1/I0
    invR    = 1.0 / R_ratio

    # denom = t + (r/c) * (1/R)
    denom_general = t + (r/c) * invR
    # tiny-z limit: denom→t + 2/a
    denom_tiny    = t + 2.0/a
    denom         = torch.where(z<1e-6, denom_tiny, denom_general)

    # -- radial speed and direction --
    v_rad = r / denom                       # scalar speed
    # avoid zero-division:
    norm_x = torch.sqrt(norm2) + eps
    dir_x  = x / norm_x

    # continuous radial field
    v_cont = x / denom#dir_x * v_rad

    # -- enforce |v| ≤ c at the boundary layer --
    threshold = c*t - eps*t
    mask      = norm_x >= threshold        # at/outside light cone
    boundary  = dir_x * c
    v_final   = torch.where(mask, boundary, v_cont)

    return v_final.to(dtype=torch.float32)

