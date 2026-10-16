"""Device selection, with Apple Silicon support."""
import torch

def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("device must be one of auto, cpu, or mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch installation.")
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")
