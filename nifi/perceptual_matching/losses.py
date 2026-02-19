"""Perceptual Matching objective components from the NiFi paper."""

from __future__ import annotations

from nifi.losses.perceptual import DISTSLoss, LPIPSLoss, ReconstructionLossBundle


class PerceptualMatchingLossBundle(ReconstructionLossBundle):
    """Paper-aligned name for the combined ``l2 + lpips (+ dists)`` losses."""


__all__ = ["LPIPSLoss", "DISTSLoss", "PerceptualMatchingLossBundle"]
