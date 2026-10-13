import numpy as np
import torch

from .analysis import lj_energy_batch


def langevin_push(
    samples,
    n_steps=100,
    step_size=1e-5,
    temperature=0.1
):

    beta = 1.0 / temperature

    x = samples.clone().detach()

    sqrt_2eps = np.sqrt(2.0 * step_size)

    for step in range(n_steps):

        x.requires_grad_(True)

        # LJ energy for all configurations
        energy = lj_energy_batch(x)

        # Gradient for all configurations
        grad_U = torch.autograd.grad(
            energy.sum(),
            x
        )[0]

        # Force
        force = -grad_U

        # Thermal noise
        noise = torch.randn_like(x)

        # Remove center-of-mass motion
        noise = noise - noise.mean(
            dim=1,
            keepdim=True
        )

        with torch.no_grad():

            x = (
                x
                + step_size * beta * force
                + sqrt_2eps * noise
            )

            # Keep COM = 0
            x = x - x.mean(
                dim=1,
                keepdim=True
            )

    return x.detach()