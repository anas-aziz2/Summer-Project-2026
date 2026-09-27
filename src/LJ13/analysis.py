import numpy as np
import torch


def lj_energy_batch(coords):
    """Lennard-Jones energy for a batch of configs.
    coords shape: [batch, N, 3]. Returns shape: [batch]."""

    diff = coords[:, :, None, :] - coords[:, None, :, :]
    r = torch.norm(diff, dim=-1)

    i, j = torch.triu_indices(
        coords.shape[1],
        coords.shape[1],
        offset=1,
        device=coords.device,
    )

    r_pairs = r[:, i, j]

    inv_r6 = (1.0 / r_pairs) ** 6
    inv_r12 = inv_r6**2

    energy = 4 * (inv_r12 - inv_r6)

    return energy.sum(dim=1)


def pairwise_distances_batch(configs):
    """All pairwise distances across a batch of configs, flattened into one array.
    configs shape: [batch, N, 3]. Returns a 1D numpy array of all distances."""

    diff = configs[:, :, None, :] - configs[:, None, :, :]
    r = torch.norm(diff, dim=-1)

    N = configs.shape[1]
    i, j = torch.triu_indices(N, N, offset=1, device=configs.device)

    r_pairs = r[:, i, j]

    return r_pairs.reshape(-1).cpu().numpy()


def effective_temperature(mean_energy, U_min, n_dof):
    """Estimate effective temperature from mean potential energy,
    using equipartition: U - U_min = (n_dof / 2) * T."""
    return (mean_energy - U_min) / (n_dof / 2)