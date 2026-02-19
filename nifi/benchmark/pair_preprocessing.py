"""Prepare benchmark datasets into NiFi clean/degraded pair layout."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from nifi.artifact_synthesis import (
    ArtifactSynthesisCompressionConfig,
    ProxyArtifactSynthesisCompressor,
    list_scene_images,
)
from nifi.benchmark.registry import PAPER_EXTREME_RATES_LAMBDA, rate_folder_name

_IMAGE_PATTERNS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")


@dataclass
class BenchmarkPairPreparationConfig:
    """Configuration for benchmark pair preprocessing."""

    rates_lambda: Sequence[float] = PAPER_EXTREME_RATES_LAMBDA
    holdout_every: int = 8
    max_images_per_scene: Optional[int] = None
    compression_method: str = "proxy"
    compression_config: ArtifactSynthesisCompressionConfig = field(default_factory=ArtifactSynthesisCompressionConfig)


def _split_names(names: List[str], holdout_every: int) -> Dict[str, List[str]]:
    split = {"train": [], "test": []}
    for idx, name in enumerate(names):
        if holdout_every > 0 and idx % int(holdout_every) == 0:
            split["test"].append(name)
        else:
            split["train"].append(name)
    return split


def _candidate_scene_dirs(dataset_root: Path) -> List[Path]:
    direct = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]

    detected: List[Path] = []
    for candidate in direct:
        try:
            list_scene_images(candidate)
            detected.append(candidate)
        except Exception:
            continue

    if detected:
        return detected

    # fallback for nested archives
    for candidate in sorted(dataset_root.rglob("*")):
        if not candidate.is_dir():
            continue
        try:
            list_scene_images(candidate)
            detected.append(candidate)
        except Exception:
            continue

    unique = []
    seen = set()
    for path in detected:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def discover_scene_dirs(dataset_root: Path, scene_names: Optional[Iterable[str]] = None) -> List[Path]:
    """Find scene directories that contain image folders usable by NiFi."""
    all_scenes = _candidate_scene_dirs(dataset_root)
    if not scene_names:
        return all_scenes

    required = {name.lower(): name for name in scene_names}
    filtered = [scene for scene in all_scenes if scene.name.lower() in required]
    return sorted(filtered)


def _find_precomputed_rate_dir(
    compressed_root: Path,
    dataset_key: str,
    scene_name: str,
    rate_dir_name: str,
) -> Optional[Path]:
    candidates = [
        compressed_root / dataset_key / scene_name / rate_dir_name,
        compressed_root / scene_name / rate_dir_name,
        compressed_root / dataset_key / scene_name / "test" / rate_dir_name,
        compressed_root / scene_name / "test" / rate_dir_name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _collect_images_from_dir(path: Path) -> List[Path]:
    images: List[Path] = []
    for pattern in _IMAGE_PATTERNS:
        images.extend(sorted(path.glob(pattern)))
    return images


def _write_clean_image(clean_src: Path, clean_dst: Path) -> None:
    clean_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clean_src, clean_dst)


def _write_proxy_degraded(
    compressor: ProxyArtifactSynthesisCompressor,
    clean_src: Path,
    degraded_dst: Path,
    rate_lambda: float,
) -> None:
    degraded_dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(clean_src) as image:
        degraded = compressor.synthesize_view_artifact(image.convert("RGB"), rate_lambda)
        degraded.save(degraded_dst)


def _write_precomputed_degraded(precomputed_image: Path, degraded_dst: Path) -> None:
    degraded_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(precomputed_image, degraded_dst)


def prepare_benchmark_pairs(
    *,
    dataset_key: str,
    dataset_root: Path,
    out_root: Path,
    config: BenchmarkPairPreparationConfig,
    scene_names: Optional[Iterable[str]] = None,
    compressed_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Create NiFi pair folders for benchmark evaluation.

    Output layout:
    ``out_root/<dataset>/<scene>/<rate>/<split>/{clean,degraded}/*.png``.
    """
    scene_dirs = discover_scene_dirs(dataset_root, scene_names=scene_names)
    if not scene_dirs:
        raise RuntimeError(f"No scenes discovered under {dataset_root}")

    if config.compression_method not in {"proxy", "precomputed"}:
        raise ValueError("compression_method must be one of: proxy, precomputed")

    if config.compression_method == "precomputed" and compressed_root is None:
        raise ValueError("compressed_root is required when compression_method='precomputed'")

    compressor = ProxyArtifactSynthesisCompressor(config.compression_config)
    manifest: Dict[str, object] = {
        "dataset": dataset_key,
        "dataset_root": str(dataset_root),
        "output_root": str(out_root),
        "compression_method": config.compression_method,
        "rates_lambda": [float(rate) for rate in config.rates_lambda],
        "scenes": {},
    }

    for scene_dir in scene_dirs:
        images = list_scene_images(scene_dir)
        if config.max_images_per_scene is not None:
            images = images[: int(config.max_images_per_scene)]

        names = [f"{idx:05d}.png" for idx in range(len(images))]
        splits = _split_names(names, holdout_every=config.holdout_every)

        scene_output = out_root / dataset_key / scene_dir.name
        scene_manifest: Dict[str, object] = {
            "scene_dir": str(scene_dir),
            "num_images": len(images),
            "splits": splits,
            "rates": {},
        }

        for image_idx, clean_src in enumerate(images):
            generated_name = names[image_idx]
            for split_name, split_members in splits.items():
                if generated_name not in split_members:
                    continue
                clean_dst = scene_output / "clean_index" / split_name / generated_name
                _write_clean_image(clean_src, clean_dst)

        for rate_lambda in config.rates_lambda:
            rate_name = rate_folder_name(rate_lambda)
            scene_manifest["rates"][rate_name] = {"train": 0, "test": 0}

            precomputed_images: List[Path] = []
            if config.compression_method == "precomputed":
                rate_dir = _find_precomputed_rate_dir(
                    compressed_root=compressed_root,
                    dataset_key=dataset_key,
                    scene_name=scene_dir.name,
                    rate_dir_name=rate_name,
                )
                if rate_dir is None:
                    raise FileNotFoundError(
                        f"Could not find precomputed degraded views for scene={scene_dir.name} rate={rate_name}"
                    )
                precomputed_images = _collect_images_from_dir(rate_dir)
                if len(precomputed_images) < len(images):
                    raise RuntimeError(
                        f"Not enough precomputed images in {rate_dir}: {len(precomputed_images)} < {len(images)}"
                    )

            for image_idx, clean_src in enumerate(images):
                generated_name = names[image_idx]
                for split_name, split_members in splits.items():
                    if generated_name not in split_members:
                        continue

                    clean_src_idx = scene_output / "clean_index" / split_name / generated_name
                    clean_dst = scene_output / rate_name / split_name / "clean" / generated_name
                    degraded_dst = scene_output / rate_name / split_name / "degraded" / generated_name

                    _write_clean_image(clean_src_idx, clean_dst)
                    if config.compression_method == "proxy":
                        _write_proxy_degraded(compressor, clean_src_idx, degraded_dst, rate_lambda=float(rate_lambda))
                    else:
                        _write_precomputed_degraded(precomputed_images[image_idx], degraded_dst)

                    scene_manifest["rates"][rate_name][split_name] += 1

        clean_index = scene_output / "clean_index"
        if clean_index.exists():
            shutil.rmtree(clean_index)

        manifest["scenes"][scene_dir.name] = scene_manifest

    manifest_path = out_root / dataset_key / "pairs_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


__all__ = [
    "BenchmarkPairPreparationConfig",
    "discover_scene_dirs",
    "prepare_benchmark_pairs",
]
