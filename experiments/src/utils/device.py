"""Device selection helpers for MPS, CUDA, or CPU."""

import torch


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
