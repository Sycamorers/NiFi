# NiFi Paper Alignment (arXiv:2602.04549)

This repository now exposes paper-aligned modules and documentation:

- Workflow map: `docs/workflow_module_mapping.md`
- Formula mapping: `docs/equation_to_code_mapping.md`
- Verification checks: `docs/implementation_verification.md`

## Paper-Aligned Code Packages
- `nifi/artifact_synthesis/`
- `nifi/artifact_restoration/`
- `nifi/restoration_distribution_matching/`
- `nifi/perceptual_matching/`
- `nifi/benchmark/`

## Equation Coverage
Equations (1)-(8) are mapped to code in:
- `nifi/restoration_distribution_matching/objectives.py`
- `nifi/artifact_restoration/diffusion_trajectory.py`
- `nifi/artifact_restoration/model.py`
- `scripts/train_nifi.py`

## Benchmark Protocol
Sec. 4.2 protocol support is provided through:
- `scripts/download_benchmark_data.py`
- `scripts/prepare_benchmark_pairs.py`
- `scripts/eval_benchmark_nifi.py`

## Verification Entry Points
- `python scripts/verify_paper_implementation.py`
- `python -m pytest -q tests/test_paper_alignment.py`
