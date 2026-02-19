# NiFi Workflow-to-Code Mapping

This repository is organized to mirror the paper's pipeline blocks in Fig. 2.

## Paper Block: Artifact Synthesis
- Purpose: Simulate low-rate 3DGS artifacts (prune/quantize/entropy-code effects) and produce paired data.
- Core modules:
  - `nifi/artifact_synthesis/gode_pruning.py`
  - `nifi/artifact_synthesis/compression_simulation.py`
  - `nifi/artifact_synthesis/pair_dataset.py`
- Main scripts:
  - `scripts/build_3dgs_and_compress.py`
  - `scripts/render_pairs.py`

## Paper Block: Artifact Restoration
- Purpose: One-step restoration in latent space with `phi_minus` over a frozen diffusion backbone.
- Core modules:
  - `nifi/artifact_restoration/model.py`
  - `nifi/artifact_restoration/diffusion_trajectory.py`
- Main scripts:
  - `scripts/train_nifi.py`
  - `scripts/eval_nifi.py`

## Paper Block: Restoration Distribution Matching
- Purpose: KL-style score matching + ground-truth direction guidance + `phi_plus` diffusion objective.
- Core modules:
  - `nifi/restoration_distribution_matching/objectives.py`
- Main scripts:
  - `scripts/train_nifi.py`
  - `scripts/verify_paper_implementation.py`

## Paper Block: Perceptual Matching
- Purpose: Optimize perceptual quality with `l2 + LPIPS (+ DISTS)` and report LPIPS/DISTS.
- Core modules:
  - `nifi/perceptual_matching/losses.py`
  - `nifi/perceptual_matching/metrics.py`
- Main scripts:
  - `scripts/train_nifi.py`
  - `scripts/eval_nifi.py`
  - `scripts/eval_benchmark_nifi.py`

## Paper Block: Benchmark Evaluation (Sec. 4.2)
- Purpose: Evaluate LPIPS/DISTS on Mip-NeRF360, Tanks & Temples, DeepBlending at rates `lambda in {0.1, 0.5, 1.0}`.
- Core modules:
  - `nifi/benchmark/registry.py`
  - `nifi/benchmark/download.py`
  - `nifi/benchmark/pair_preprocessing.py`
  - `nifi/benchmark/paired_dataset.py`
  - `nifi/benchmark/evaluation_protocol.py`
- Main scripts:
  - `scripts/download_benchmark_data.py`
  - `scripts/prepare_benchmark_pairs.py`
  - `scripts/eval_benchmark_nifi.py`

## Pipeline Summary
1. Download scene data (`scripts/download_data.py`, `scripts/download_benchmark_data.py`).
2. Synthesize artifacts (`scripts/build_3dgs_and_compress.py`).
3. Build clean/degraded pairs (`scripts/render_pairs.py`, `scripts/prepare_benchmark_pairs.py`).
4. Train NiFi (`scripts/train_nifi.py`).
5. Evaluate (`scripts/eval_nifi.py`, `scripts/eval_benchmark_nifi.py`).
