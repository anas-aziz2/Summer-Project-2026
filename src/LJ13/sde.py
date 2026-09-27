import numpy as np
from scipy.spatial.distance import pdist


def estimate_sigma_max(flat_configs, n_subset=25000, seed=None):
    """flat_configs: (N, D) numpy array — e.g. centered configs reshaped to N x (n_atoms*3)."""
    rng = np.random.default_rng(seed)
    n = flat_configs.shape[0]
    idx = rng.permutation(n)[:n_subset]
    subset = flat_configs[idx]
    return pdist(subset).max()


class VESDE:
    def __init__(self, sigma_max, sigma_min=0.02):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def g(self, t):
        sigma = self.sigma(t)
        return sigma * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))