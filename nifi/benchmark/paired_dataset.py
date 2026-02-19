"""Benchmark pair dataset for NiFi paper evaluation protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class BenchmarkPairSample:
    """One benchmark sample containing clean/degraded images and metadata."""

    dataset: str
    scene: str
    rate: str
    name: str
    clean_path: Path
    degraded_path: Path
    prompt: str


class NiFiBenchmarkPairDataset(Dataset):
    """Dataset layout: ``root/<dataset>/<scene>/<rate>/<split>/{clean,degraded}/*.png``."""

    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = 256,
        max_samples: Optional[int] = None,
        allowed_rates: Optional[List[str]] = None,
        allowed_datasets: Optional[List[str]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.allowed_rates = set(allowed_rates) if allowed_rates else None
        self.allowed_datasets = set(allowed_datasets) if allowed_datasets else None

        self.samples = self._scan_samples()
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]

        self.tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _scan_samples(self) -> List[BenchmarkPairSample]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Benchmark data root not found: {self.data_root}")

        items: List[BenchmarkPairSample] = []
        for dataset_dir in sorted([p for p in self.data_root.iterdir() if p.is_dir()]):
            if self.allowed_datasets and dataset_dir.name not in self.allowed_datasets:
                continue

            for scene_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
                prompt = self._read_scene_prompt(scene_dir)
                for rate_dir in sorted([p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("rate_")]):
                    if self.allowed_rates and rate_dir.name not in self.allowed_rates:
                        continue
                    split_dir = rate_dir / self.split
                    clean_dir = split_dir / "clean"
                    degraded_dir = split_dir / "degraded"
                    if not clean_dir.exists() or not degraded_dir.exists():
                        continue

                    for clean_path in sorted(clean_dir.glob("*.png")):
                        degraded_path = degraded_dir / clean_path.name
                        if not degraded_path.exists():
                            continue
                        items.append(
                            BenchmarkPairSample(
                                dataset=dataset_dir.name,
                                scene=scene_dir.name,
                                rate=rate_dir.name,
                                name=clean_path.stem,
                                clean_path=clean_path,
                                degraded_path=degraded_path,
                                prompt=prompt,
                            )
                        )

        if not items:
            raise RuntimeError(
                f"No benchmark pairs found under {self.data_root}. "
                "Run scripts/prepare_benchmark_pairs.py first."
            )

        return items

    @staticmethod
    def _read_scene_prompt(scene_dir: Path) -> str:
        txt_path = scene_dir / "prompt.txt"
        if txt_path.exists():
            try:
                return txt_path.read_text().strip()
            except Exception:
                return ""

        json_path = scene_dir / "prompt.json"
        if json_path.exists():
            try:
                with json_path.open("r") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    for key in ("prompt", "text", "caption"):
                        value = payload.get(key)
                        if isinstance(value, str):
                            return value.strip()
            except Exception:
                return ""

        return ""

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        clean = self.tf(self._load_rgb(sample.clean_path))
        degraded = self.tf(self._load_rgb(sample.degraded_path))

        return {
            "dataset": sample.dataset,
            "scene": sample.scene,
            "rate": sample.rate,
            "name": sample.name,
            "clean": clean,
            "degraded": degraded,
            "prompt": sample.prompt,
        }


__all__ = ["BenchmarkPairSample", "NiFiBenchmarkPairDataset"]
