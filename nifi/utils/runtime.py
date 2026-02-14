from typing import Dict

import torch


def configure_runtime(runtime_cfg: Dict[str, object]) -> None:
    deterministic = bool(runtime_cfg.get("deterministic", False))
    cudnn_benchmark = bool(runtime_cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(runtime_cfg.get("allow_tf32", True))

    torch.backends.cudnn.benchmark = cudnn_benchmark and not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        pass



def resolve_device(runtime_cfg: Dict[str, object]) -> torch.device:
    requested = str(runtime_cfg.get("device", "cuda")).lower()
    device_id = int(runtime_cfg.get("device_id", 0))

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available. Set runtime.device=cpu to run on CPU.")
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")

    return torch.device("cpu")



def get_runtime_defaults() -> Dict[str, object]:
    return {
        "device": "cuda",
        "device_id": 0,
        "deterministic": False,
        "allow_tf32": True,
        "cudnn_benchmark": True,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "non_blocking": True,
        "empty_cache_before_eval": False,
    }
