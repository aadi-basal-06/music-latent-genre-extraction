"""YAML configuration loading and validation."""
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for key in ("data", "training", "model"):
        if key not in config:
            raise ValueError(f"Configuration is missing '{key}'.")
    if config["training"].get("batch_size", 0) <= 0:
        raise ValueError("training.batch_size must be positive.")
    return config
