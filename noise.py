"""
Cosine noise schedule and forward diffusion process.
"""

import math
import torch


def cosine_schedule(T: int, s: float = 0.008) -> dict:
    """
    Cosine schedule from "Improved DDPM" (Nichol & Dhariwal 2021).
    Gentler than linear — less aggressive corruption at high t,
    which helps for small molecules where structure matters even at moderate noise.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]                          # ᾱ_t, shape (T+1,)
    alpha_bar = torch.clamp(alpha_bar, min=1e-5)

    alpha = alpha_bar[1:] / alpha_bar[:-1]        # α_t = ᾱ_t / ᾱ_{t-1}
    beta = 1 - alpha                              # β_t

    # Posterior variance for reverse step
    alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])  # (T+1,)
    posterior_variance = beta * (1 - alpha_bar_prev[:-1]) / (1 - alpha_bar[1:])

    return {
        'alpha_bar':          alpha_bar.float(),           # (T+1,)
        'alpha':              alpha.float(),                # (T,)
        'beta':               beta.float(),                 # (T,)
        'posterior_variance': posterior_variance.float(),   # (T,)
    }


def q_sample(coords: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor):
    """
    Forward process: corrupt clean coords to noise level t.
    coords: (N, 2)
    t:      (1,) or scalar — timestep index
    Returns noisy coords and the noise that was added.
    """
    noise = torch.randn_like(coords)
    abar = alpha_bar[t].to(coords.device)          # scalar
    noisy = coords * abar.sqrt() + noise * (1 - abar).sqrt()
    return noisy, noise


def predict_coords_from_noise(noisy: torch.Tensor, pred_noise: torch.Tensor,
                               t: int, schedule: dict) -> torch.Tensor:
    """Reconstruct coords_0 estimate from noisy coords and predicted noise."""
    abar = schedule['alpha_bar'][t]
    return (noisy - pred_noise * (1 - abar).sqrt()) / abar.sqrt()


def p_sample_step_x0(noisy: torch.Tensor, x0_pred: torch.Tensor,
                     t: int, schedule: dict) -> torch.Tensor:
    """One reverse step using x0 prediction (EGNN output) instead of noise."""
    abar = schedule['alpha_bar'][t]
    # Convert x0 prediction to implied noise, then reuse standard reverse step
    pred_noise = (noisy - x0_pred * abar.sqrt()) / (1 - abar).sqrt().clamp(min=1e-8)
    return p_sample_step(noisy, pred_noise, t, schedule)


def p_sample_step(noisy: torch.Tensor, pred_noise: torch.Tensor,
                  t: int, schedule: dict) -> torch.Tensor:
    """
    One reverse diffusion step: coords_t → coords_{t-1}.
    Adds posterior noise unless t=0 (last step is deterministic).
    """
    alpha     = schedule['alpha'][t - 1]
    alpha_bar = schedule['alpha_bar'][t]
    beta      = schedule['beta'][t - 1]
    post_var  = schedule['posterior_variance'][t - 1]

    # DDPM reverse mean
    coef = beta / (1 - alpha_bar).sqrt()
    mean = (noisy - coef * pred_noise) / alpha.sqrt()

    if t == 1:
        return mean
    noise = torch.randn_like(noisy)
    return mean + post_var.sqrt() * noise
