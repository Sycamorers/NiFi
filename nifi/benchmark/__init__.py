"""Benchmark datasets and protocol utilities for NiFi."""

from nifi.benchmark.download import (
    copy_local_tree,
    download_mipnerf360,
    download_tandt_deepblending_bundle,
)
from nifi.benchmark.evaluation_protocol import (
    PAPER_NIFI_RESULTS,
    aggregate_benchmark_records,
    compare_with_paper,
)
from nifi.benchmark.pair_preprocessing import (
    BenchmarkPairPreparationConfig,
    discover_scene_dirs,
    prepare_benchmark_pairs,
)
from nifi.benchmark.paired_dataset import NiFiBenchmarkPairDataset
from nifi.benchmark.registry import (
    BENCHMARK_DATASETS,
    PAPER_EVAL_METRICS,
    PAPER_EXTREME_RATES_LAMBDA,
    list_supported_datasets,
    rate_folder_name,
)

__all__ = [
    "copy_local_tree",
    "download_mipnerf360",
    "download_tandt_deepblending_bundle",
    "PAPER_NIFI_RESULTS",
    "aggregate_benchmark_records",
    "compare_with_paper",
    "BenchmarkPairPreparationConfig",
    "discover_scene_dirs",
    "prepare_benchmark_pairs",
    "NiFiBenchmarkPairDataset",
    "BENCHMARK_DATASETS",
    "PAPER_EVAL_METRICS",
    "PAPER_EXTREME_RATES_LAMBDA",
    "list_supported_datasets",
    "rate_folder_name",
]
