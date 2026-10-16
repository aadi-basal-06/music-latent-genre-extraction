"""Genre-stratified, track-level splitting."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

def track_level_split(metadata: list[dict], seed: int = 42,
                      ratios: tuple[float, float, float] = (0.75, 0.125, 0.125)) -> dict[str, list[str]]:
    """Split unique track IDs per genre so no audio track leaks across partitions."""
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    groups: dict[int, set[str]] = defaultdict(set)
    for row in metadata:
        groups[int(row["label"])].add(str(row["track_id"]))
    rng = np.random.default_rng(seed)
    result = {"train": [], "val": [], "test": []}
    for tracks in groups.values():
        ordered = np.array(sorted(tracks)); rng.shuffle(ordered)
        n = len(ordered); n_train = int(n * ratios[0]); n_val = int(n * ratios[1])
        result["train"] += ordered[:n_train].tolist()
        result["val"] += ordered[n_train:n_train + n_val].tolist()
        result["test"] += ordered[n_train + n_val:].tolist()
    return result

def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))

def load_splits(path: str | Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())
