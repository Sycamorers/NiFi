"""Perceptual benchmark metrics used in NiFi evaluation."""

from __future__ import annotations

from nifi.metrics.perceptual_metrics import PerceptualMetrics, aggregate_scene_metrics


class PerceptualMatchingMetrics(PerceptualMetrics):
    """Paper-aligned alias for LPIPS and DISTS metric computation."""


__all__ = ["PerceptualMatchingMetrics", "aggregate_scene_metrics"]
