"""Evaluation protocol helpers for NiFi benchmark reporting."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np

from nifi.benchmark.registry import PAPER_EVAL_METRICS


PAPER_NIFI_RESULTS = {
    "mipnerf360": {
        "rate_0.100": {"lpips_after": 0.178, "dists_after": 0.109},
        "rate_0.500": {"lpips_after": 0.235, "dists_after": 0.133},
        "rate_1.000": {"lpips_after": 0.265, "dists_after": 0.153},
    },
    "tanks_temples": {
        "rate_0.100": {"lpips_after": 0.128, "dists_after": 0.076},
        "rate_0.500": {"lpips_after": 0.180, "dists_after": 0.095},
        "rate_1.000": {"lpips_after": 0.212, "dists_after": 0.109},
    },
    "deepblending": {
        "rate_0.100": {"lpips_after": 0.133, "dists_after": 0.101},
        "rate_0.500": {"lpips_after": 0.180, "dists_after": 0.131},
        "rate_1.000": {"lpips_after": 0.218, "dists_after": 0.156},
    },
}


def _mean(items: Iterable[float]) -> float:
    values = list(items)
    return float(np.mean(values)) if values else float("nan")


def aggregate_benchmark_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-image benchmark records by dataset, scene, and rate."""
    per_dataset = defaultdict(list)
    per_scene = defaultdict(list)
    per_dataset_rate = defaultdict(list)

    for record in records:
        dataset = str(record["dataset"])
        scene = str(record["scene"])
        rate = str(record["rate"])
        per_dataset[dataset].append(record)
        per_scene[(dataset, scene)].append(record)
        per_dataset_rate[(dataset, rate)].append(record)

    def _aggregate_group(group: List[Dict[str, object]]) -> Dict[str, float]:
        return {
            "lpips_before": _mean(float(item["lpips_before"]) for item in group),
            "lpips_after": _mean(float(item["lpips_after"]) for item in group),
            "dists_before": _mean(float(item["dists_before"]) for item in group),
            "dists_after": _mean(float(item["dists_after"]) for item in group),
            "num_images": len(group),
        }

    aggregate = _aggregate_group(records) if records else {}

    dataset_summary = {dataset: _aggregate_group(items) for dataset, items in sorted(per_dataset.items())}

    dataset_rate_summary = {
        f"{dataset}/{rate}": _aggregate_group(items)
        for (dataset, rate), items in sorted(per_dataset_rate.items())
    }

    scene_summary = {
        f"{dataset}/{scene}": _aggregate_group(items)
        for (dataset, scene), items in sorted(per_scene.items())
    }

    return {
        "metrics": list(PAPER_EVAL_METRICS),
        "aggregate": aggregate,
        "per_dataset": dataset_summary,
        "per_dataset_rate": dataset_rate_summary,
        "per_scene": scene_summary,
    }


def compare_with_paper(summary: Dict[str, object]) -> Dict[str, object]:
    """Compare benchmark aggregate metrics against paper Table 1 NiFi values."""
    per_dataset_rate = summary.get("per_dataset_rate", {}) if isinstance(summary, dict) else {}

    comparison: Dict[str, object] = {}
    for dataset, rate_table in PAPER_NIFI_RESULTS.items():
        dataset_comp: Dict[str, object] = {}
        for rate_name, expected in rate_table.items():
            key = f"{dataset}/{rate_name}"
            measured = per_dataset_rate.get(key)
            if not isinstance(measured, dict):
                dataset_comp[rate_name] = {
                    "status": "missing",
                    "expected": expected,
                    "measured": None,
                }
                continue

            measured_lpips = float(measured["lpips_after"])
            measured_dists = float(measured["dists_after"])
            dataset_comp[rate_name] = {
                "status": "available",
                "expected": expected,
                "measured": {
                    "lpips_after": measured_lpips,
                    "dists_after": measured_dists,
                },
                "delta": {
                    "lpips_after": measured_lpips - float(expected["lpips_after"]),
                    "dists_after": measured_dists - float(expected["dists_after"]),
                },
            }
        comparison[dataset] = dataset_comp

    return comparison


__all__ = [
    "PAPER_NIFI_RESULTS",
    "aggregate_benchmark_records",
    "compare_with_paper",
]
