"""Processed-array datasets and loaders."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class GenreDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if self.features.ndim != 2 or self.features.shape[1] != 500:
            raise ValueError("features must have shape (N, 500)")
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels have different lengths")
    def __len__(self) -> int: return len(self.labels)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: return self.features[index], self.labels[index]

def load_processed_split(processed_dir: str | Path, split_path: str | Path, split: str) -> GenreDataset:
    root = Path(processed_dir); metadata = json.loads((root / "metadata.json").read_text())
    tracks = set(json.loads(Path(split_path).read_text())[split])
    mask = np.array([row["track_id"] in tracks for row in metadata])
    return GenreDataset(np.load(root / "features.npy")[mask], np.load(root / "labels.npy")[mask])
