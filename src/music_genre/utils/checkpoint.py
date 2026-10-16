"""Portable training checkpoint helpers."""
from pathlib import Path
from typing import Any
import torch

def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, best_metric: float, config: dict[str, Any], seed: int) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "best_metric": best_metric, "config": config, "seed": seed}, path)

def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
