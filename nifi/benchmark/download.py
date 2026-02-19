"""Benchmark dataset download helpers for NiFi evaluation protocol."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm

from nifi.benchmark.registry import (
    MIPNERF360_BASE_URL,
    MIPNERF360_SCENES,
    TANDT_DEEPBLENDING_ARCHIVE_URL,
)


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download ``url`` to ``out_path`` with a progress bar and skip-on-cache."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with out_path.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a ZIP archive to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(out_dir)


def download_mipnerf360(
    out_root: Path,
    scenes: Optional[Iterable[str]] = None,
    remove_zip: bool = False,
) -> None:
    """Download Mip-NeRF360 scene archives."""
    scenes_set = list(scenes) if scenes is not None else list(MIPNERF360_SCENES)

    for scene in scenes_set:
        if scene not in MIPNERF360_SCENES:
            raise ValueError(f"Unknown Mip-NeRF360 scene: {scene}")

        url = f"{MIPNERF360_BASE_URL}/{scene}.zip"
        zip_path = out_root / "mipnerf360" / f"{scene}.zip"
        download_file(url, zip_path)
        extract_zip(zip_path, out_root / "mipnerf360")
        if remove_zip:
            zip_path.unlink(missing_ok=True)


def download_tandt_deepblending_bundle(
    out_root: Path,
    remove_zip: bool = False,
) -> Path:
    """Download the shared Tanks&Temples + DeepBlending benchmark bundle."""
    zip_path = out_root / "tanks_temples_deepblending" / "tandt_db.zip"
    download_file(TANDT_DEEPBLENDING_ARCHIVE_URL, zip_path)
    extract_root = out_root / "tanks_temples_deepblending"
    extract_zip(zip_path, extract_root)
    if remove_zip:
        zip_path.unlink(missing_ok=True)
    return extract_root


def copy_local_tree(source: Path, destination: Path, overwrite: bool = False) -> None:
    """Copy an already-downloaded dataset tree into the benchmark root."""
    if not source.exists():
        raise FileNotFoundError(f"Local benchmark source does not exist: {source}")

    if destination.exists():
        if not overwrite:
            return
        shutil.rmtree(destination)

    shutil.copytree(source, destination)


__all__ = [
    "download_file",
    "extract_zip",
    "download_mipnerf360",
    "download_tandt_deepblending_bundle",
    "copy_local_tree",
]
