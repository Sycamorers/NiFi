"""Artifact synthesis blocks from the NiFi paper workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from PIL import Image

from nifi.gs.compressor import (
    CompressionConfig,
    HACPPWrapper,
    Proxy3DGSCompressor,
    list_scene_images,
)


@dataclass
class ArtifactSynthesisCompressionConfig:
    """Configuration for simulated compression artifacts."""

    jpeg_quality_min: int = 8
    jpeg_quality_max: int = 55
    downsample_min: int = 2
    downsample_max: int = 8

    def to_legacy(self) -> CompressionConfig:
        """Convert to the legacy compressor configuration object."""
        return CompressionConfig(
            jpeg_quality_min=int(self.jpeg_quality_min),
            jpeg_quality_max=int(self.jpeg_quality_max),
            downsample_min=int(self.downsample_min),
            downsample_max=int(self.downsample_max),
        )


class ProxyArtifactSynthesisCompressor:
    """Proxy implementation of NiFi's Artifact Synthesis stage."""

    def __init__(self, config: ArtifactSynthesisCompressionConfig):
        self.config = config
        self._compressor = Proxy3DGSCompressor(config.to_legacy())

    def synthesize_view_artifact(self, clean_image: Image.Image, rate_lambda: float) -> Image.Image:
        """Simulate low-rate 3DGS rendering artifacts for one image."""
        return self._compressor.degrade_image(clean_image, float(rate_lambda))

    def synthesize_scene_artifacts(
        self,
        scene_dir: Path,
        rates_lambda: Sequence[float],
        out_dir: Path,
        holdout_every: int = 8,
        max_images: Optional[int] = None,
    ) -> Dict[str, object]:
        """Generate paired clean/degraded images for training or evaluation."""
        return self._compressor.build_scene_artifacts(
            scene_dir=scene_dir,
            rates=rates_lambda,
            out_dir=out_dir,
            holdout_every=holdout_every,
            max_images=max_images,
        )


class HACPPArtifactSynthesisWrapper(HACPPWrapper):
    """Thin wrapper naming HAC++ integration as an Artifact Synthesis block."""


__all__ = [
    "ArtifactSynthesisCompressionConfig",
    "ProxyArtifactSynthesisCompressor",
    "HACPPArtifactSynthesisWrapper",
    "list_scene_images",
]
