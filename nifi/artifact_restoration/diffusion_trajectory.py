"""Diffusion trajectory utilities for NiFi Artifact Restoration."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from nifi.diffusion.schedule import SigmaSchedule


class DiffusionTrajectorySigmaSchedule(SigmaSchedule):
    """Paper-aligned naming for the ``sigma_t`` schedule in Eq. (2)-(4)."""

    def forward_diffusion_eq2(
        self,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Eq. (2): ``x_t = (1 - sigma_t)x + sigma_t eps``."""
        return self.q_sample(clean_latents, timesteps, noise=noise)

    @staticmethod
    def inverse_diffusion_step_eq3(
        noisy_latents_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        sigma_t: torch.Tensor,
        sigma_t_minus_1: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Eq. (3): ``x_{t-1} = x_t - (sigma_{t-1} - sigma_t) eps_theta``."""
        delta = (sigma_t_minus_1 - sigma_t).view(-1, 1, 1, 1)
        return noisy_latents_t - delta * predicted_noise

    @staticmethod
    def score_proxy_eq4(noisy_latents_t: torch.Tensor, predicted_noise: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """Apply Eq. (4) score proxy ``s(x_t) = x_t - sigma_t eps``."""
        return SigmaSchedule.score_from_eps(noisy_latents_t, predicted_noise, sigma_t)


__all__ = ["DiffusionTrajectorySigmaSchedule"]
