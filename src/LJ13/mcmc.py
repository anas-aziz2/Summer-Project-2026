import numpy as np


def read_xyz(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0])

    coords = []
    for line in lines[2:]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(coords), n_atoms


def lj_energy(positions, epsilon=1.0, lj_sigma=1.0):
    N = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    i, j = np.triu_indices(N, k=1)
    inv6 = (lj_sigma / r[i, j]) ** 6
    return np.sum(4 * epsilon * (inv6**2 - inv6))


def atom_energy(positions, i, epsilon=1.0, lj_sigma=1.0):
    others = np.delete(positions, i, axis=0)
    r = np.linalg.norm(positions[i] - others, axis=1)
    inv6 = (lj_sigma / r) ** 6
    return np.sum(4 * epsilon * (inv6**2 - inv6))


def run_chain(
    initial_coords,
    n_steps,
    seed,
    temperature=0.10,
    max_displacement=0.043,
    burn_in=20000,
    stride=4000,
    epsilon=1.0,
    lj_sigma=1.0,
    k_b=1.0,
):

    rng = np.random.default_rng(seed)

    coords = initial_coords.copy()
    n_atoms = coords.shape[0]
    current_energy = lj_energy(coords, epsilon, lj_sigma)

    energy_history = np.empty(n_steps)
    configs = []
    accepted = 0

    for step in range(n_steps):
        atom = rng.integers(n_atoms)
        displacement = rng.uniform(-max_displacement, max_displacement, size=3)

        new_coords = coords.copy()
        new_coords[atom] += displacement

        deltaE = (
            atom_energy(new_coords, atom, epsilon, lj_sigma)
            - atom_energy(coords, atom, epsilon, lj_sigma)
        )

        if deltaE <= 0 or rng.random() < np.exp(-deltaE / (k_b * temperature)):
            coords = new_coords
            current_energy += deltaE
            accepted += 1

        energy_history[step] = current_energy

        if step > burn_in and step % stride == 0:
            configs.append(coords.copy())

    return np.array(configs), energy_history, accepted / n_steps