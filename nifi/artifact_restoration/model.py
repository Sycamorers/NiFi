"""Artifact Restoration model components from the NiFi workflow."""

from __future__ import annotations

from typing import List

import torch

from nifi.diffusion.model import DiffusionConfig, FrozenLDMWithNiFiAdapters


class ArtifactRestorationDiffusionConfig(DiffusionConfig):
    """Paper-aligned alias for diffusion configuration."""


class FrozenBackboneArtifactRestorationModel(FrozenLDMWithNiFiAdapters):
    """Frozen LDM backbone with trainable ``phi_minus`` and ``phi_plus`` adapters."""

    def encode_image_to_latent_space(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space with the VAE encoder ``E``."""
        return self.encode_images(images)

    def decode_latent_to_image_space(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent tensors back to image space with decoder ``D``."""
        return self.decode_latents(latents)


def project_to_intermediate_diffusion_step_eq2(
    model: FrozenBackboneArtifactRestorationModel,
    degraded_latents: torch.Tensor,
    t0: int,
    stochastic: bool,
) -> torch.Tensor:
    """Project degraded latents to an intermediate diffusion timestep ``t0`` via Eq. (2)."""
    batch_size = degraded_latents.shape[0]
    timestep = torch.full((batch_size,), int(t0), device=degraded_latents.device, dtype=torch.long)
    noise = torch.randn_like(degraded_latents) if stochastic else torch.zeros_like(degraded_latents)
    projected_latents, _ = model.q_sample(degraded_latents, timestep, noise=noise)
    return projected_latents


def artifact_restoration_one_step_eq7(
    model: FrozenBackboneArtifactRestorationModel,
    degraded_latents: torch.Tensor,
    prompts: List[str],
    t0: int,
    stochastic: bool,
) -> torch.Tensor:
    """Apply Eq. (7): one-step restoration using ``phi_minus`` at timestep ``t0``."""
    batch_size = degraded_latents.shape[0]
    timestep = torch.full((batch_size,), int(t0), device=degraded_latents.device, dtype=torch.long)

    noise = torch.randn_like(degraded_latents) if stochastic else torch.zeros_like(degraded_latents)
    projected_latents, _ = model.q_sample(degraded_latents, timestep, noise=noise)
    predicted_noise = model.predict_eps(projected_latents, timestep, prompts, adapter_type="minus", train_mode=stochastic)

    sigma_t0 = model.sigma(timestep).view(-1, 1, 1, 1)
    restored_latents = projected_latents - sigma_t0 * predicted_noise
    return restored_latents


__all__ = [
    "ArtifactRestorationDiffusionConfig",
    "FrozenBackboneArtifactRestorationModel",
    "project_to_intermediate_diffusion_step_eq2",
    "artifact_restoration_one_step_eq7",
]
