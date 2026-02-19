"""Benchmark dataset registry for NiFi paper evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


MIPNERF360_BASE_URL = "https://storage.googleapis.com/gresearch/refraw360"
TANDT_DEEPBLENDING_ARCHIVE_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
)

PAPER_EVAL_METRICS: Tuple[str, str] = ("lpips", "dists")
PAPER_EXTREME_RATES_LAMBDA: Tuple[float, float, float] = (0.1, 0.5, 1.0)


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    """Metadata for one benchmark dataset used in paper evaluation."""

    key: str
    paper_name: str
    download_urls: Tuple[str, ...]
    default_scenes: Tuple[str, ...]
    notes: str


MIPNERF360_SCENES: Tuple[str, ...] = (
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
)

TANKS_AND_TEMPLES_SCENES: Tuple[str, ...] = (
    "train",
    "truck",
)

DEEPBLENDING_SCENES: Tuple[str, ...] = (
    "drjohnson",
    "playroom",
)


BENCHMARK_DATASETS: Dict[str, BenchmarkDatasetSpec] = {
    "mipnerf360": BenchmarkDatasetSpec(
        key="mipnerf360",
        paper_name="Mip-NeRF360",
        download_urls=tuple(f"{MIPNERF360_BASE_URL}/{scene}.zip" for scene in MIPNERF360_SCENES),
        default_scenes=MIPNERF360_SCENES,
        notes="Official Mip-NeRF360 scene archives.",
    ),
    "tanks_temples": BenchmarkDatasetSpec(
        key="tanks_temples",
        paper_name="Tanks & Temples",
        download_urls=(TANDT_DEEPBLENDING_ARCHIVE_URL,),
        default_scenes=TANKS_AND_TEMPLES_SCENES,
        notes="Scenes are provided inside tandt_db.zip from the 3DGS benchmark assets.",
    ),
    "deepblending": BenchmarkDatasetSpec(
        key="deepblending",
        paper_name="DeepBlending",
        download_urls=(TANDT_DEEPBLENDING_ARCHIVE_URL,),
        default_scenes=DEEPBLENDING_SCENES,
        notes="Scenes are provided inside tandt_db.zip from the 3DGS benchmark assets.",
    ),
}


def rate_folder_name(rate_lambda: float) -> str:
    """Convert paper rate lambda to the canonical folder name."""
    return f"rate_{float(rate_lambda):.3f}"


def list_supported_datasets() -> List[str]:
    """Return benchmark dataset keys in deterministic order."""
    return sorted(BENCHMARK_DATASETS.keys())


__all__ = [
    "BENCHMARK_DATASETS",
    "BenchmarkDatasetSpec",
    "PAPER_EVAL_METRICS",
    "PAPER_EXTREME_RATES_LAMBDA",
    "rate_folder_name",
    "list_supported_datasets",
]
