"""Paired clean/degraded dataset for the Artifact Synthesis output."""

from __future__ import annotations

from typing import List, Optional

from torch.utils.data import DataLoader

from nifi.data.builders import build_paired_dataloader
from nifi.data.paired_dataset import PairedImageDataset


class NiFiArtifactPairDataset(PairedImageDataset):
    """Paper-aligned alias for the paired training dataset."""


def build_artifact_pair_dataloader(
    data_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: Optional[int] = None,
    allowed_rates: Optional[List[str]] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Build a dataloader over Artifact Synthesis clean/degraded pairs."""
    return build_paired_dataloader(
        data_root=data_root,
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        max_samples=max_samples,
        allowed_rates=allowed_rates,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


__all__ = ["NiFiArtifactPairDataset", "build_artifact_pair_dataloader"]
