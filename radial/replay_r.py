import torch
from kac_utils_radial import *
class OneDTrajectoryBuffer:
    def __init__(self, a, T, buffer_size, max_samples, device):
        self.a = a
        self.T = T
        self.buffer_size = buffer_size
        self.max_samples = max_samples
        self.device = device

        self.s_k = torch.empty(buffer_size, max_samples, device=device)
        self.cum_sums = torch.empty(buffer_size, max_samples, device=device)

        self.fill_buffer()

    def _generate_trajectory(self):
        s_k = torch.distributions.Exponential(self.a).sample((self.max_samples,)).to(self.device)
        cum_sums = torch.cumsum(s_k, dim=0)
        return s_k, cum_sums

    def fill_buffer(self):
        for i in range(self.buffer_size):
            self.s_k[i], self.cum_sums[i] = self._generate_trajectory()

    def partial_update(self, frac=0.1):
        n_update = int(self.buffer_size * frac)
        update_indices = torch.randperm(self.buffer_size, device=self.device)[:n_update]
        # Generate new draws for all indices at once:
        new_s_k = torch.distributions.Exponential(self.a).sample((n_update, self.max_samples)).to(self.device)
        new_cum = torch.cumsum(new_s_k, dim=1)
        self.s_k[update_indices] = new_s_k
        self.cum_sums[update_indices] = new_cum

    def sample(self, num_samples):
        idx = torch.randint(0, self.buffer_size, (num_samples,), device=self.device)
        return self.s_k[idx], self.cum_sums[idx]  # shape: (num_samples, max_samples)

class TwoTierTrajectoryBuffer:
    """
    A two‐tier buffer for sampling telegrapher‐process trajectories:
      - main_buffer:   stores trajectories for random t ∈ [0, T]
      - T_buffer:      stores trajectories *only* for t = T
    Both buffers hold (s_k, cum_sums) pairs with shape (buffer_size, max_samples).
    """

    def __init__(self, a, T, main_buffer_size, T_buffer_size, max_samples, device):
        """
        Args:
          a, T: same as before (telegrapher parameters)
          main_buffer_size:  number of rows in the “random‐t” buffer
          T_buffer_size:     number of rows in the “t=T” buffer
          max_samples:       the maximum number of exponential jumps (int(2 a T)+50)
          device:            "cpu" or "cuda"
        """
        self.a = a
        self.T = T
        self.max_samples = max_samples
        self.device = device

        # Main buffer (random t)
        self.main_buffer = OneDTrajectoryBuffer(
            a=a, T=T, buffer_size=main_buffer_size,
            max_samples=max_samples, device=device
        )

        # T_buffer (only for t = T)
        self.T_buffer = OneDTrajectoryBuffer(
            a=a, T=T, buffer_size=T_buffer_size,
            max_samples=max_samples, device=device
        )

    def sample(self, num_samples):
        """
        Sample num_samples rows from the main buffer (for random‐t draws).
        Returns (s_k, cum_sums) each of shape (num_samples, max_samples).
        """
        s_k, cum_sums = self.main_buffer.sample(num_samples)
        return s_k, cum_sums

    def sample_T(self, num_samples):
        """
        Sample num_samples rows from the T_buffer (for t = T draws).
        Returns (s_k, cum_sums) each of shape (num_samples, max_samples).
        """
        s_k_T, cum_sums_T = self.T_buffer.sample(num_samples)
        return s_k_T, cum_sums_T

    def partial_update(self, frac_main=0.1, frac_T=0.1):
        """
        Refresh a fraction of each sub‐buffer’s entries by drawing new Exponentials.
          - frac_main: fraction (0 < frac_main <= 1) of main_buffer to replace
          - frac_T:    fraction (0 < frac_T <= 1) of T_buffer to replace
        """
        # Update main_buffer
        self.main_buffer.partial_update(frac=frac_main)
        # Update T_buffer
        self.T_buffer.partial_update(frac=frac_T)
