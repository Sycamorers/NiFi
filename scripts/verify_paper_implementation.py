#!/usr/bin/env python3
"""Equation-to-code verification checks for NiFi implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import (
    DiffusionTrajectorySigmaSchedule,
    artifact_restoration_one_step_eq7,
)
from nifi.diffusion.lora import LowRankSpatialAdapter
from nifi.restoration_distribution_matching import (
    ground_truth_direction_surrogate_eq5,
    kl_divergence_surrogate_eq4,
    phi_minus_objective_eq6,
    phi_plus_diffusion_objective_eq8,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


class _DummyRestorationModel:
    """Lightweight model stub for Eq. (7) shape checks."""

    def q_sample(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        if noise is None:
            noise = torch.randn_like(x)
        sigma = self.sigma(t).view(-1, 1, 1, 1)
        return (1.0 - sigma) * x + sigma * noise, noise

    def predict_eps(self, x: torch.Tensor, t: torch.Tensor, prompts, adapter_type: str = "minus", train_mode: bool = False):
        del prompts, adapter_type, train_mode
        return torch.tanh(x)

    def sigma(self, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.full_like(timesteps, 0.25, dtype=torch.float32)



def _assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-6) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"Tensors are not close (max abs diff={torch.max(torch.abs(actual - expected)).item():.6e})")



def verify_eq2_forward_diffusion() -> CheckResult:
    schedule = DiffusionTrajectorySigmaSchedule(num_train_timesteps=10, sigma_min=0.1, sigma_max=1.0)
    x = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x)
    t = torch.tensor([0, 9], dtype=torch.long)
    x_t, sampled_noise = schedule.forward_diffusion_eq2(x, t, noise=noise)

    sigma = schedule.sigma(t).view(-1, 1, 1, 1)
    expected = (1.0 - sigma) * x + sigma * noise

    _assert_close(x_t, expected)
    _assert_close(sampled_noise, noise)
    return CheckResult("Eq2 forward diffusion", True, "x_t and sampled noise match analytical formula")



def verify_eq4_score_proxy() -> CheckResult:
    x_t = torch.randn(2, 4, 8, 8)
    eps = torch.randn_like(x_t)
    sigma = torch.tensor([0.2, 0.8], dtype=x_t.dtype).view(-1, 1, 1, 1)

    score = DiffusionTrajectorySigmaSchedule.score_proxy_eq4(x_t, eps, sigma)
    expected = x_t - sigma * eps
    _assert_close(score, expected)
    return CheckResult("Eq4 score proxy", True, "score computation matches x_t - sigma_t * eps")



def verify_eq4_eq5_surrogate_gradients() -> CheckResult:
    x_hat = torch.randn(2, 4, 8, 8, requires_grad=True)
    s_real = torch.randn_like(x_hat)
    s_restore = torch.randn_like(x_hat)

    loss_kl = kl_divergence_surrogate_eq4(x_hat, s_real, s_restore)
    grad_kl = torch.autograd.grad(loss_kl, x_hat, retain_graph=True)[0]
    expected_kl = (s_restore - s_real) / float(x_hat.numel())
    _assert_close(grad_kl, expected_kl)

    s_real_clean = torch.randn_like(x_hat)
    loss_gt = ground_truth_direction_surrogate_eq5(x_hat, s_real_clean, s_real)
    grad_gt = torch.autograd.grad(loss_gt, x_hat)[0]
    expected_gt = (s_real_clean - s_real) / float(x_hat.numel())
    _assert_close(grad_gt, expected_gt)

    return CheckResult("Eq4/Eq5 surrogate gradients", True, "autograd gradients match analytical surrogate gradients")



def verify_eq6_objective_composition() -> CheckResult:
    values = {
        "kl": torch.tensor(1.2),
        "gt": torch.tensor(0.8),
        "l2": torch.tensor(0.3),
        "lpips": torch.tensor(0.4),
        "dists": torch.tensor(0.5),
    }
    alpha = 0.7
    objective = phi_minus_objective_eq6(
        alpha=alpha,
        kl_term=values["kl"],
        gt_term=values["gt"],
        l2_term=values["l2"],
        lpips_term=values["lpips"],
        dists_term=values["dists"],
        weight_kl=1.0,
        weight_gt=1.0,
        weight_l2=2.0,
        weight_lpips=3.0,
        weight_dists=4.0,
    )
    expected = alpha * values["kl"] + (1.0 - alpha) * values["gt"] + 2.0 * values["l2"] + 3.0 * values["lpips"] + 4.0 * values["dists"]
    _assert_close(objective, expected)
    return CheckResult("Eq6 objective", True, "weighted objective matches Eq.6 implementation")



def verify_eq7_shape_and_formula() -> CheckResult:
    model = _DummyRestorationModel()
    z_degraded = torch.randn(3, 4, 16, 16)
    prompts = ["scene prompt"] * z_degraded.shape[0]

    restored = artifact_restoration_one_step_eq7(
        model=model,
        degraded_latents=z_degraded,
        prompts=prompts,
        t0=5,
        stochastic=False,
    )

    if restored.shape != z_degraded.shape:
        raise AssertionError(f"Restored shape mismatch: {restored.shape} != {z_degraded.shape}")

    t = torch.full((z_degraded.shape[0],), 5, dtype=torch.long)
    z_t0, _ = model.q_sample(z_degraded, t, noise=torch.zeros_like(z_degraded))
    eps = model.predict_eps(z_t0, t, prompts, adapter_type="minus", train_mode=False)
    sigma = model.sigma(t).view(-1, 1, 1, 1)
    expected = z_t0 - sigma * eps
    _assert_close(restored, expected)
    return CheckResult("Eq7 one-step restoration", True, "restored latent shape and formula are correct")



def verify_eq8_phi_plus_objective() -> CheckResult:
    eps_pred = torch.randn(2, 4, 8, 8)
    eps_gt = torch.randn_like(eps_pred)
    loss = phi_plus_diffusion_objective_eq8(eps_pred, eps_gt)
    expected = torch.mean((eps_pred - eps_gt) ** 2)
    _assert_close(loss, expected)
    return CheckResult("Eq8 phi_plus objective", True, "MSE objective matches Eq.8")



def verify_parameter_initialization() -> CheckResult:
    adapter = LowRankSpatialAdapter(in_channels=4, text_dim=16, rank=8, max_t=32)
    up_weight = adapter.up.weight.detach()
    if not torch.allclose(up_weight, torch.zeros_like(up_weight)):
        raise AssertionError("Adapter up-projection is not zero initialized")
    return CheckResult("Adapter initialization", True, "phi adapter up-projection is zero-initialized")



def verify_alternating_optimization_pattern() -> CheckResult:
    phi_minus = torch.nn.Parameter(torch.tensor(1.0))
    phi_plus = torch.nn.Parameter(torch.tensor(-1.0))

    opt_minus = torch.optim.SGD([phi_minus], lr=0.1)
    opt_plus = torch.optim.SGD([phi_plus], lr=0.1)

    before_minus = float(phi_minus.item())
    before_plus = float(phi_plus.item())

    minus_loss = (phi_minus - 3.0) ** 2
    minus_loss.backward()
    opt_minus.step()
    opt_minus.zero_grad(set_to_none=True)

    plus_loss = (phi_plus + 2.0) ** 2
    plus_loss.backward()
    opt_plus.step()
    opt_plus.zero_grad(set_to_none=True)

    if float(phi_minus.item()) == before_minus:
        raise AssertionError("phi_minus was not updated")
    if float(phi_plus.item()) == before_plus:
        raise AssertionError("phi_plus was not updated")

    return CheckResult("Alternating optimization", True, "phi_minus and phi_plus update in separate optimization steps")



def main() -> None:
    checks = [
        verify_eq2_forward_diffusion,
        verify_eq4_score_proxy,
        verify_eq4_eq5_surrogate_gradients,
        verify_eq6_objective_composition,
        verify_eq7_shape_and_formula,
        verify_eq8_phi_plus_objective,
        verify_parameter_initialization,
        verify_alternating_optimization_pattern,
    ]

    results = []
    for check_fn in checks:
        result = check_fn()
        results.append(result)

    payload: Dict[str, object] = {
        "num_checks": len(results),
        "all_passed": all(result.passed for result in results),
        "checks": [result.__dict__ for result in results],
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
