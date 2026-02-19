#!/usr/bin/env python3
"""Evaluate NiFi checkpoint on benchmark protocol datasets (Sec. 4.2)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nifi.artifact_restoration import (
    ArtifactRestorationDiffusionConfig,
    FrozenBackboneArtifactRestorationModel,
    artifact_restoration_one_step_eq7,
)
from nifi.benchmark import (
    NiFiBenchmarkPairDataset,
    aggregate_benchmark_records,
    compare_with_paper,
    list_supported_datasets,
)
from nifi.perceptual_matching import PerceptualMatchingMetrics
from nifi.utils.checkpoint import load_checkpoint
from nifi.utils.config import load_config
from nifi.utils.logging import get_logger
from nifi.utils.runtime import configure_runtime, get_runtime_defaults, resolve_device
from nifi.utils.seed import set_seed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NiFi on benchmark datasets with LPIPS/DISTS")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True, help="Prepared benchmark pairs root")
    parser.add_argument("--out", type=str, required=True, help="JSON output path")

    parser.add_argument("--config", type=str, default=None, help="Optional config override")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--datasets", nargs="*", default=None, choices=list_supported_datasets())
    parser.add_argument("--rates", nargs="*", default=None, help="Optional rate folders, e.g. rate_0.100")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--save_restored", action="store_true")
    parser.add_argument("--restored_dir", type=str, default=None)
    return parser.parse_args()



def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    args = parse_args()
    logger = get_logger("nifi.eval_benchmark")

    checkpoint = load_checkpoint(args.ckpt, map_location="cpu")

    if args.config:
        cfg = load_config(args.config)
    elif checkpoint.get("extra", {}).get("config"):
        cfg = checkpoint["extra"]["config"]
    else:
        raise ValueError("Could not find config in checkpoint. Pass --config explicitly.")

    runtime_cfg = get_runtime_defaults()
    runtime_cfg.update(cfg.get("runtime", {}))

    set_seed(int(cfg.get("seed", 42)), deterministic=bool(runtime_cfg.get("deterministic", False)))
    configure_runtime(runtime_cfg)
    device = resolve_device(runtime_cfg)
    non_blocking = bool(runtime_cfg.get("non_blocking", True))

    mixed_precision = cfg["train"].get("mixed_precision", "fp16")
    amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    use_amp = device.type == "cuda" and mixed_precision in {"bf16", "fp16"}

    model_cfg = ArtifactRestorationDiffusionConfig(
        model_name_or_path=cfg["model"]["pretrained_model_name_or_path"],
        num_train_timesteps=int(cfg["diffusion"]["num_train_timesteps"]),
        lora_rank=int(cfg["model"]["lora_rank"]),
        guidance_scale=float(cfg["model"]["guidance_scale"]),
        prompt_dropout=float(cfg["model"]["prompt_dropout"]),
        max_token_length=int(cfg["model"]["max_token_length"]),
        vae_scaling_factor=float(cfg["model"]["vae_scaling_factor"]),
    )

    model = FrozenBackboneArtifactRestorationModel(
        model_cfg,
        device=device,
        dtype=amp_dtype if use_amp else torch.float32,
    )
    model.phi_minus.load_state_dict(checkpoint["phi_minus"])
    model.phi_plus.load_state_dict(checkpoint["phi_plus"])
    model.freeze_backbone()
    model.eval()

    dataset = NiFiBenchmarkPairDataset(
        data_root=args.data_root,
        split=args.split,
        image_size=int(cfg.get("image_size", 256) or 256),
        max_samples=args.max_samples,
        allowed_rates=args.rates,
        allowed_datasets=args.datasets,
    )

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": bool(runtime_cfg.get("pin_memory", True)),
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        dataloader_kwargs["persistent_workers"] = bool(runtime_cfg.get("persistent_workers", True))
        dataloader_kwargs["prefetch_factor"] = int(runtime_cfg.get("prefetch_factor", 4))

    dataloader = DataLoader(**dataloader_kwargs)

    metrics = PerceptualMatchingMetrics(device=device)
    t0 = (
        int(cfg["diffusion"]["num_train_timesteps"] - 1)
        if cfg["diffusion"].get("t_ablation_full", False)
        else int(cfg["diffusion"]["t0"])
    )

    restored_dir = Path(args.restored_dir) if args.restored_dir else None
    if args.save_restored and restored_dir is None:
        restored_dir = Path(args.out).parent / "restored"
    if restored_dir is not None:
        restored_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []

    for batch in tqdm(dataloader, desc="eval_benchmark"):
        clean = batch["clean"].to(device, non_blocking=non_blocking)
        degraded = batch["degraded"].to(device, non_blocking=non_blocking)

        prompts = list(batch["prompt"])
        datasets = list(batch["dataset"])
        scenes = list(batch["scene"])
        rates = list(batch["rate"])
        names = list(batch["name"])

        with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            degraded_latents = model.encode_images(degraded)
            restored_latents = artifact_restoration_one_step_eq7(
                model,
                degraded_latents,
                prompts,
                t0=t0,
                stochastic=False,
            )
            restored = model.decode_latents(restored_latents)

            lpips_before = metrics.lpips(degraded.float(), clean.float())
            lpips_after = metrics.lpips(restored.float(), clean.float())
            dists_before = metrics.dists(degraded.float(), clean.float())
            dists_after = metrics.dists(restored.float(), clean.float())

        if args.save_restored:
            from torchvision.utils import save_image

            restored_img = ((restored.clamp(-1.0, 1.0) + 1.0) * 0.5).detach().cpu()
            for index, image in enumerate(restored_img):
                output_path = restored_dir / datasets[index] / scenes[index] / rates[index]
                output_path.mkdir(parents=True, exist_ok=True)
                save_image(image, output_path / f"{names[index]}.png")

        for index in range(clean.shape[0]):
            records.append(
                {
                    "dataset": datasets[index],
                    "scene": scenes[index],
                    "rate": rates[index],
                    "name": names[index],
                    "lpips_before": float(lpips_before[index].item()),
                    "lpips_after": float(lpips_after[index].item()),
                    "dists_before": float(dists_before[index].item()),
                    "dists_after": float(dists_after[index].item()),
                }
            )

    summary = aggregate_benchmark_records(records)
    paper_comparison = compare_with_paper(summary)

    payload = {
        "checkpoint": str(args.ckpt),
        "split": args.split,
        "num_samples": len(records),
        "records": records,
        "summary": summary,
        "paper_comparison": paper_comparison,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = out_path.with_suffix(".csv")
    save_csv(csv_path, records)

    aggregate = summary.get("aggregate", {})
    logger.info(
        "Benchmark LPIPS %.4f -> %.4f | DISTS %.4f -> %.4f",
        float(aggregate.get("lpips_before", float("nan"))),
        float(aggregate.get("lpips_after", float("nan"))),
        float(aggregate.get("dists_before", float("nan"))),
        float(aggregate.get("dists_after", float("nan"))),
    )
    logger.info("Wrote benchmark JSON: %s", out_path)
    logger.info("Wrote benchmark CSV : %s", csv_path)


if __name__ == "__main__":
    main()
