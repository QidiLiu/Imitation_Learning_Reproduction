from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(in_seed: int, in_deterministic_torch: bool = False) -> None:
    """Seed Python, Numpy and PyTorch. Call once per process, before model creation."""
    os.environ["PYTHONHASHSEED"] = str(in_seed)
    random.seed(in_seed)
    np.random.seed(in_seed)
    torch.manual_seed(in_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(in_seed)
    if in_deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(in_requested: str = "cuda") -> torch.device:
    """Resolve the training device. CUDA is the default; CPU is explict only."""
    if in_requested == "auto":
        in_requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(in_requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. Install a CUDA-enabled "
            "PyTorch build, or pass device='cpu' only for smoke tests."
        )
    return device
