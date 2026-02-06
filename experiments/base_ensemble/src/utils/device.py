"""Device selection helpers for MPS, CUDA, or CPU."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device: MPS, CUDA, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Return a human-readable device label for logs."""
    if device.type == "mps":
        return "MPS (Metal Performance Shaders)"
    elif device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    return "CPU"
