import numpy as np
import torch


def predictor_corrector_sampler(
    model,
    sde,
    n_atoms=13,
    n_dim=3,
    n_samples=1000,
    n_steps=1000,
    n_corrector=1,
    snr=0.16,
):
    model.eval()

    x = torch.randn(n_samples, n_atoms, n_dim)
    x = x - x.mean(dim=1, keepdim=True)
    x = x * sde.sigma_max

    dt = 1.0 / n_steps
    sqrt_dt = np.sqrt(dt)

    with torch.no_grad():
        for i in range(n_steps):
            t = 1 - i / n_steps
            t_tensor = torch.full((n_samples, 1), t)

            sigma = sde.sigma(t_tensor)
            log_sigma = torch.log(sigma)
            g_val = sde.g(t_tensor).view(-1, 1, 1)

            f_pred = model(x, log_sigma)
            score = f_pred / sigma.view(-1, 1, 1)

            if torch.isnan(score).any():
                print(f"NaN in score at step {i}, t={t:.4f}, sigma={sigma[0].item():.4f}")
                break

            noise = torch.randn_like(x)
            noise = noise - noise.mean(dim=1, keepdim=True)

            x = x + (g_val**2) * score * dt + g_val * sqrt_dt * noise

            if torch.isnan(x).any():
                print(f"NaN in x (predictor) at step {i}")
                break

            for _ in range(n_corrector):
                f_pred = model(x, log_sigma)
                score = f_pred / sigma.view(-1, 1, 1)

                noise = torch.randn_like(x)
                noise = noise - noise.mean(dim=1, keepdim=True)

                score_norm = torch.norm(score.reshape(n_samples, -1), dim=1).mean()
                noise_norm = torch.norm(noise.reshape(n_samples, -1), dim=1).mean()

                if score_norm.item() == 0.0 or torch.isnan(score_norm):
                    print(f"score_norm degenerate at step {i}: {score_norm.item()}")
                    break

                eps = 2 * (snr * noise_norm / score_norm) ** 2
                x = x + eps * score + torch.sqrt(2 * eps) * noise

                if torch.isnan(x).any():
                    print(f"NaN in x (corrector) at step {i}")
                    break

    return x