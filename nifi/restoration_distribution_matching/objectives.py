"""Restoration Distribution Matching objectives from NiFi Eq. (4)-(8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nifi.losses.distillation import kl_score_surrogate_loss, score_guidance_surrogate_loss


def kl_divergence_surrogate_eq4(
    restored_latents: torch.Tensor,
    score_real_distribution: torch.Tensor,
    score_restoration_distribution: torch.Tensor,
) -> torch.Tensor:
    """Eq. (4) surrogate whose gradient matches ``s_restore - s_real``."""
    return kl_score_surrogate_loss(
        restored_latents=restored_latents,
        score_real=score_real_distribution,
        score_restore=score_restoration_distribution,
    )


def ground_truth_direction_surrogate_eq5(
    restored_latents: torch.Tensor,
    score_real_clean: torch.Tensor,
    score_real_restored: torch.Tensor,
) -> torch.Tensor:
    """Eq. (5) ground-truth direction surrogate ``s_real(x) - s_real(x_hat)``."""
    return score_guidance_surrogate_loss(
        restored_latents=restored_latents,
        score_clean_real=score_real_clean,
        score_restored_real=score_real_restored,
    )


def phi_minus_objective_eq6(
    *,
    alpha: float,
    kl_term: torch.Tensor,
    gt_term: torch.Tensor,
    l2_term: torch.Tensor,
    lpips_term: torch.Tensor,
    dists_term: torch.Tensor,
    weight_kl: float = 1.0,
    weight_gt: float = 1.0,
    weight_l2: float = 1.0,
    weight_lpips: float = 1.0,
    weight_dists: float = 0.0,
) -> torch.Tensor:
    """Compose Eq. (6) objective with the paper's weighted terms."""
    alpha = float(alpha)
    return (
        alpha * float(weight_kl) * kl_term
        + (1.0 - alpha) * float(weight_gt) * gt_term
        + float(weight_l2) * l2_term
        + float(weight_lpips) * lpips_term
        + float(weight_dists) * dists_term
    )


def phi_plus_diffusion_objective_eq8(
    predicted_noise: torch.Tensor,
    sampled_noise: torch.Tensor,
) -> torch.Tensor:
    """Eq. (8): ``L_phi_plus = l2(eps, eps_{theta,phi+}(x_hat_t, t))``."""
    return F.mse_loss(predicted_noise, sampled_noise)


__all__ = [
    "kl_divergence_surrogate_eq4",
    "ground_truth_direction_surrogate_eq5",
    "phi_minus_objective_eq6",
    "phi_plus_diffusion_objective_eq8",
]
