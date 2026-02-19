# Implementation Verification Report

This report records explicit checks between the paper and code.

## Verified Against Equations
Command used:
```bash
python scripts/verify_paper_implementation.py
```

Checks passed:
- Eq. (2) forward diffusion formula.
- Eq. (4) score proxy formula.
- Eq. (4)/(5) surrogate gradient direction.
- Eq. (6) objective composition and weights.
- Eq. (7) one-step restoration formula and tensor shape preservation.
- Eq. (8) diffusion MSE objective.
- Adapter initialization (`phi` up-projection initialized to zero).
- Alternating optimization pattern (`phi_minus` and `phi_plus` updated in separate steps).

Additional unit tests:
```bash
python -m pytest -q tests/test_paper_alignment.py
```

## Tensor Shapes and Dimensions
- Clean/degraded image tensors: `[B, 3, H, W]`.
- Latent tensors: `[B, C, h, w]` (from VAE encoder).
- Timesteps: `[B]` (`torch.long`).
- `sigma_t` broadcasting: reshaped to `[B, 1, 1, 1]`.
- One-step restoration output shape equals degraded latent shape.

## Loss Definitions
- `L_phi_minus` (Eq. 6): KL surrogate + GT-direction surrogate + `l2 + lpips (+ dists)`.
- `L_phi_plus` (Eq. 8): MSE between sampled noise and predicted noise.
- Benchmark metrics: LPIPS and DISTS (lower is better), aligned with Sec. 4.2/Table 1.

## Corrections Applied During Refactor
- Introduced paper-aligned modules for each major process block:
  - `nifi/artifact_synthesis/`
  - `nifi/artifact_restoration/`
  - `nifi/restoration_distribution_matching/`
  - `nifi/perceptual_matching/`
  - `nifi/benchmark/`
- Updated training/evaluation scripts to call paper-named functions and objectives directly.
- Added benchmark download/preprocess/evaluation scripts for Sec. 4.2 protocol.

## Known Fidelity Gaps vs Paper (Documented)
- Backbone default remains a lightweight SD-compatible model in `configs/default.yaml`; paper uses SD3.
- Artifact synthesis default is proxy-based unless HAC++ outputs are provided.
- Prompt extraction via Qwen2.5-VL is not fully automated in this repo.

These gaps are expected to affect exact metric matching versus paper Table 1.
