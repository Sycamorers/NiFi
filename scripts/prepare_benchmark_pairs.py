#!/usr/bin/env python3
"""Preprocess benchmark scenes into NiFi clean/degraded pair layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_synthesis import ArtifactSynthesisCompressionConfig
from nifi.benchmark import (
    BenchmarkPairPreparationConfig,
    list_supported_datasets,
    prepare_benchmark_pairs,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NiFi benchmark pairs from downloaded datasets")
    parser.add_argument("--dataset", type=str, required=True, choices=list_supported_datasets())
    parser.add_argument("--dataset_root", type=str, required=True, help="Root folder containing scene directories")
    parser.add_argument("--out", type=str, required=True, help="Output pair root, e.g. pairs/benchmarks")
    parser.add_argument("--scene", nargs="*", default=None, help="Optional scene-name subset")

    parser.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--holdout_every", type=int, default=8)
    parser.add_argument("--max_images_per_scene", type=int, default=None)

    parser.add_argument("--compression_method", type=str, default="proxy", choices=["proxy", "precomputed"])
    parser.add_argument("--compressed_root", type=str, default=None, help="Required when --compression_method precomputed")

    parser.add_argument("--jpeg_quality_min", type=int, default=8)
    parser.add_argument("--jpeg_quality_max", type=int, default=55)
    parser.add_argument("--downsample_min", type=int, default=2)
    parser.add_argument("--downsample_max", type=int, default=8)
    return parser.parse_args()



def main() -> None:
    args = parse_args()

    compression_cfg = ArtifactSynthesisCompressionConfig(
        jpeg_quality_min=int(args.jpeg_quality_min),
        jpeg_quality_max=int(args.jpeg_quality_max),
        downsample_min=int(args.downsample_min),
        downsample_max=int(args.downsample_max),
    )

    prep_cfg = BenchmarkPairPreparationConfig(
        rates_lambda=[float(rate) for rate in args.rates],
        holdout_every=int(args.holdout_every),
        max_images_per_scene=args.max_images_per_scene,
        compression_method=args.compression_method,
        compression_config=compression_cfg,
    )

    manifest = prepare_benchmark_pairs(
        dataset_key=args.dataset,
        dataset_root=Path(args.dataset_root),
        out_root=Path(args.out),
        config=prep_cfg,
        scene_names=args.scene,
        compressed_root=Path(args.compressed_root) if args.compressed_root else None,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
