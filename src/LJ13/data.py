import numpy as np
import torch


def load_configs(path, n_atoms=13, n_dim=3):
    """Load a .npy file of flattened configurations and reshape to (N, n_atoms, n_dim)."""
    configs = np.load(path)
    configs = configs.reshape(-1, n_atoms, n_dim)
    return configs


def center_configs(configs):
    """Subtract the center of mass from each configuration."""
    com = configs.mean(axis=1, keepdims=True)
    return configs - com


def configs_to_tensor(configs, dtype=torch.float32):
    return torch.tensor(configs, dtype=dtype)