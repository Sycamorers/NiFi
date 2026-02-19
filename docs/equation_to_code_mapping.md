# Formula-to-Code Mapping (NiFi Paper)

This mapping links Eq. (1)-(8) in the paper to implementation modules and functions.

## Equation (1)
Description: KL objective between restored and real image distributions.
File: `nifi/restoration_distribution_matching/objectives.py`
Function: `kl_divergence_surrogate_eq4()`
Key code snippet:
```python
return kl_score_surrogate_loss(
    restored_latents=restored_latents,
    score_real=score_real_distribution,
    score_restore=score_restoration_distribution,
)
```

## Equation (2)
Description: Forward diffusion parameterization `x_t = (1 - sigma_t)x + sigma_t eps`.
File: `nifi/artifact_restoration/diffusion_trajectory.py`
Function: `DiffusionTrajectorySigmaSchedule.forward_diffusion_eq2()`
Key code snippet:
```python
return self.q_sample(clean_latents, timesteps, noise=noise)
```

## Equation (3)
Description: One-step inverse diffusion update `x_{t-1} = x_t - (sigma_{t-1} - sigma_t) eps_theta`.
File: `nifi/artifact_restoration/diffusion_trajectory.py`
Function: `DiffusionTrajectorySigmaSchedule.inverse_diffusion_step_eq3()`
Key code snippet:
```python
delta = (sigma_t_minus_1 - sigma_t).view(-1, 1, 1, 1)
return noisy_latents_t - delta * predicted_noise
```

## Equation (4)
Description: Score proxy and KL-gradient surrogate for distribution matching.
File: `nifi/artifact_restoration/diffusion_trajectory.py`
Function: `DiffusionTrajectorySigmaSchedule.score_proxy_eq4()`
Key code snippet:
```python
return SigmaSchedule.score_from_eps(noisy_latents_t, predicted_noise, sigma_t)
```

File: `nifi/restoration_distribution_matching/objectives.py`
Function: `kl_divergence_surrogate_eq4()`
Key code snippet:
```python
return kl_score_surrogate_loss(
    restored_latents=restored_latents,
    score_real=score_real_distribution,
    score_restore=score_restoration_distribution,
)
```

## Equation (5)
Description: Ground-truth direction guidance term.
File: `nifi/restoration_distribution_matching/objectives.py`
Function: `ground_truth_direction_surrogate_eq5()`
Key code snippet:
```python
return score_guidance_surrogate_loss(
    restored_latents=restored_latents,
    score_clean_real=score_real_clean,
    score_restored_real=score_real_restored,
)
```

## Equation (6)
Description: Final `phi_minus` objective combining matching and perceptual losses.
File: `nifi/restoration_distribution_matching/objectives.py`
Function: `phi_minus_objective_eq6()`
Key code snippet:
```python
return (
    alpha * weight_kl * kl_term
    + (1.0 - alpha) * weight_gt * gt_term
    + weight_l2 * l2_term
    + weight_lpips * lpips_term
    + weight_dists * dists_term
)
```

## Equation (7)
Description: Inference-time one-step artifact restoration at intermediate `t0`.
File: `nifi/artifact_restoration/model.py`
Function: `artifact_restoration_one_step_eq7()`
Key code snippet:
```python
projected_latents, _ = model.q_sample(degraded_latents, timestep, noise=noise)
predicted_noise = model.predict_eps(projected_latents, timestep, prompts, adapter_type="minus")
restored_latents = projected_latents - sigma_t0 * predicted_noise
```

## Equation (8)
Description: `phi_plus` diffusion objective (noise MSE).
File: `nifi/restoration_distribution_matching/objectives.py`
Function: `phi_plus_diffusion_objective_eq8()`
Key code snippet:
```python
return F.mse_loss(predicted_noise, sampled_noise)
```

## Training Loop Link
File: `scripts/train_nifi.py`
Functions: `main()`, `evaluate()`
How equations are used:
- Eq. (2): `model.q_sample(...)`
- Eq. (4)/(5): `kl_divergence_surrogate_eq4(...)`, `ground_truth_direction_surrogate_eq5(...)`
- Eq. (6): `phi_minus_objective_eq6(...)`
- Eq. (8): `phi_plus_diffusion_objective_eq8(...)`
- Eq. (7) (eval path): `artifact_restoration_one_step_eq7(...)`
