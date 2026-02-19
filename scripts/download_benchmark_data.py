#!/usr/bin/env python3
"""Download benchmark datasets used by NiFi evaluation (Sec. 4.2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.benchmark import (
    list_supported_datasets,
    download_mipnerf360,
    download_tandt_deepblending_bundle,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NiFi benchmark datasets (Mip-NeRF360, Tanks&Temples, DeepBlending)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", *list_supported_datasets()],
        help="Dataset key to download, or 'all'.",
    )
    parser.add_argument("--out", type=str, required=True, help="Output root, e.g. data/benchmarks")
    parser.add_argument("--mip_scenes", nargs="*", default=None, help="Optional Mip-NeRF360 scene subset")
    parser.add_argument("--remove_zip", action="store_true", help="Delete ZIP archives after extraction")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    targets: List[str]
    if args.dataset == "all":
        targets = list_supported_datasets()
    else:
        targets = [args.dataset]

    summary = {
        "out_root": str(out_root),
        "downloaded": [],
    }

    if "mipnerf360" in targets:
        download_mipnerf360(out_root=out_root, scenes=args.mip_scenes, remove_zip=bool(args.remove_zip))
        summary["downloaded"].append("mipnerf360")

    if "tanks_temples" in targets or "deepblending" in targets:
        download_tandt_deepblending_bundle(out_root=out_root, remove_zip=bool(args.remove_zip))
        if "tanks_temples" in targets:
            summary["downloaded"].append("tanks_temples")
        if "deepblending" in targets:
            summary["downloaded"].append("deepblending")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
