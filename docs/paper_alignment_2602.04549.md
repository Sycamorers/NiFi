# NiFi Paper Alignment Guide (arXiv:2602.04549)

This guide maps the paper equations and pipeline stages to this repo, and flags what is still repro-lite.

## 1) Equation-to-code mapping

- Eq. 2 (`x_t = (1 - sigma_t) x + sigma_t eps`): `nifi/diffusion/schedule.py`
- Eq. 7 one-step restore (`x_hat = x_tilde_t0 - sigma_t0 * eps_{theta,phi-}`): `scripts/train_nifi.py`, `scripts/eval_nifi.py`
- Eq. 4 score proxy (`s(x_t) = x_t - sigma_t * eps`): `nifi/diffusion/schedule.py`
- Eq. 4 KL surrogate for `phi_minus`: `nifi/losses/distillation.py` (`kl_score_surrogate_loss`)
- Eq. 5 ground-truth real-score guidance term: `nifi/losses/distillation.py` (`score_guidance_surrogate_loss`) and `scripts/train_nifi.py`
- Eq. 8 `phi_plus` diffusion noise MSE: `scripts/train_nifi.py`

## 2) Paper hyperparameters reflected in defaults

`configs/default.yaml` now matches the paper values where supported:

- `model.guidance_scale: 7.5`
- `model.lora_rank: 64`
- `diffusion.t0: 199`
- `train.batch_size: 4`
- `train.max_steps: 60000`
- `train.lr_phi_minus: 5e-6`
- `train.lr_phi_plus: 1e-6`
- `train.weight_decay: 1e-4`
- `train.grad_clip: 1.0`
- `loss_weights.alpha: 0.7`
- perceptual matching defaults to `L2 + LPIPS` (`dists` weight is set to `0.0`)

## 3) Prompt conditioning path

- Scene-level prompts are read from either:
  - `pairs/<scene>/prompt.txt`, or
  - `pairs/<scene>/prompt.json` (`prompt`, `text`, or `caption` key)
- Prompt dropout is applied during training, controlled by `model.prompt_dropout`.

## 4) Remaining repro-lite gaps vs the paper

These are the major gaps that still prevent an exact replication:

- Backbone is SD1/SD2-style (`CLIPTextModel + UNet2DConditionModel`), not SD3.
- Artifact synthesis default is proxy image degradation, not full GoDe/HAC++ pruning + quantization + entropy pipeline.
- Qwen2.5-VL prompt extraction is not automated in this repo.
- No built-in implementation of the paper's primitive-cardinality pruning schedule (`cmin = 4096` with GoDe-style levels).

## 5) Recommended paper-faithful run mode

1. Use `--method hacpp` in `scripts/build_3dgs_and_compress.py` and provide real HAC++ commands.
2. Train on DL3DV scenes (103-scene protocol) with scene prompts saved per scene.
3. Keep `configs/default.yaml` paper-matched optimization settings.
4. Replace current diffusion backbone module with an SD3-compatible implementation if exact backbone fidelity is required.
