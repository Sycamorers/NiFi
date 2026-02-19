"""Unit tests for equation-level paper alignment."""

from __future__ import annotations

import torch

from nifi.artifact_restoration import DiffusionTrajectorySigmaSchedule
from nifi.diffusion.lora import LowRankSpatialAdapter
from nifi.restoration_distribution_matching import (
    ground_truth_direction_surrogate_eq5,
    kl_divergence_surrogate_eq4,
    phi_minus_objective_eq6,
    phi_plus_diffusion_objective_eq8,
)


def test_eq2_forward_diffusion_formula() -> None:
    schedule = DiffusionTrajectorySigmaSchedule(num_train_timesteps=10, sigma_min=0.05, sigma_max=1.0)
    x = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x)
    t = torch.tensor([0, 9], dtype=torch.long)
    x_t, sampled_noise = schedule.forward_diffusion_eq2(x, t, noise=noise)

    sigma = schedule.sigma(t).view(-1, 1, 1, 1)
    expected = (1.0 - sigma) * x + sigma * noise
    assert torch.allclose(sampled_noise, noise)
    assert torch.allclose(x_t, expected)


def test_eq4_and_eq5_surrogate_gradients() -> None:
    x_hat = torch.randn(2, 4, 8, 8, requires_grad=True)
    s_real = torch.randn_like(x_hat)
    s_restore = torch.randn_like(x_hat)
    s_clean = torch.randn_like(x_hat)

    loss_kl = kl_divergence_surrogate_eq4(x_hat, s_real, s_restore)
    grad_kl = torch.autograd.grad(loss_kl, x_hat, retain_graph=True)[0]
    assert torch.allclose(grad_kl, (s_restore - s_real) / float(x_hat.numel()), atol=1e-6)

    loss_gt = ground_truth_direction_surrogate_eq5(x_hat, s_clean, s_real)
    grad_gt = torch.autograd.grad(loss_gt, x_hat)[0]
    assert torch.allclose(grad_gt, (s_clean - s_real) / float(x_hat.numel()), atol=1e-6)


def test_eq6_and_eq8_objectives() -> None:
    eq6 = phi_minus_objective_eq6(
        alpha=0.7,
        kl_term=torch.tensor(1.0),
        gt_term=torch.tensor(2.0),
        l2_term=torch.tensor(0.3),
        lpips_term=torch.tensor(0.4),
        dists_term=torch.tensor(0.5),
        weight_kl=1.0,
        weight_gt=1.0,
        weight_l2=2.0,
        weight_lpips=3.0,
        weight_dists=4.0,
    )
    expected_eq6 = 0.7 * 1.0 + 0.3 * 2.0 + 2.0 * 0.3 + 3.0 * 0.4 + 4.0 * 0.5
    assert torch.isclose(eq6, torch.tensor(expected_eq6, dtype=eq6.dtype))

    pred = torch.randn(2, 4, 8, 8)
    target = torch.randn_like(pred)
    eq8 = phi_plus_diffusion_objective_eq8(pred, target)
    assert torch.isclose(eq8, torch.mean((pred - target) ** 2))


def test_adapter_initialization_zero_up_projection() -> None:
    adapter = LowRankSpatialAdapter(in_channels=4, text_dim=16, rank=8, max_t=32)
    assert torch.allclose(adapter.up.weight, torch.zeros_like(adapter.up.weight))
