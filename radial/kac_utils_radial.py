import torch

if torch.cuda.is_available(): device = torch.device("cuda")
elif torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

def create_sk(a, t, T):
    """
    Sample exponential durations and return durations and cumulative sums.
    a: rate
    t: Tensor of shape (batch,)
    T: float (max time horizon)
    Returns:
      s_k : Tensor of shape (batch, max_samples)
      cum_sums : Tensor of shape (batch, max_samples)
    """
    batch_size = t.shape[0]
    max_samples = int(2 * torch.max(t) * a) + 50  # Add margin for safety
    s_k = torch.distributions.Exponential(a).sample((batch_size, max_samples)).to(device)
    cum_sums = torch.cumsum(s_k, dim=1)
    return s_k, cum_sums


def sample_trj(t, s_k, cum_sums, a, T):
    """
    Compute the signed displacement tau_t for the Kac telegraph process at time t.

    Args:
      t        : Tensor of shape (batch,)
      s_k      : Tensor of shape (batch, max_samples)
      cum_sums : Tensor of shape (batch, max_samples)
      a        : rate parameter (float)
      T        : time horizon (float)

    Returns:
      tau_t : Tensor of shape (batch,)
    """
    batch_size = t.shape[0]
    device = t.device
    max_samples = s_k.shape[1]

    # Determine number of completed jumps before time t
    mask = cum_sums < t.unsqueeze(1)  # (batch, max_samples)
    last_valid_idx = mask.sum(dim=1)  # (batch,) counts of valid samples

    # Sum of durations up to last completed jump
    sum_s_k = torch.gather(cum_sums, 1, last_valid_idx.unsqueeze(1)).squeeze(1)

    # Build index tensor for masking
    indices = torch.arange(max_samples, device=device).unsqueeze(0).expand(batch_size, max_samples)
    valid_s_k_mask = indices <= last_valid_idx.unsqueeze(1)

    # Alternating sum of durations: (-1)^k s_k
    signs = (-1) ** indices
    alt_sums = torch.cumsum(signs * s_k, dim=1) * valid_s_k_mask
    alt_sum_s_k = torch.gather(alt_sums, 1, last_valid_idx.unsqueeze(1)).squeeze(1)

    # Compute tau_t by adding the partial interval
    n = last_valid_idx
    tau_t = alt_sum_s_k + ((-1) ** n) * (t - sum_s_k)
    return tau_t


def create_sk_diff(a, t, T):
    batch_size = t.shape[0]
    max_samples = int(2 * torch.max(t) * a) + 50
    s_k = torch.distributions.Exponential(a).rsample((batch_size, max_samples)).to(device)
    cum_sums = torch.cumsum(s_k, dim=1)
    return s_k, cum_sums
