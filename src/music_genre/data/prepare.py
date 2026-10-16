"""Dataset discovery and persisted preprocessing."""
from pathlib import Path
import json
import numpy as np
from music_genre import LABEL_TO_ID
from .audio_loader import load_audio
from .segmentation import segment_audio
from .preprocessing import pool_to_features
from .splits import track_level_split, save_splits

def prepare_dataset(data_dir: str | Path, output_dir: str | Path, splits_path: str | Path, seed: int = 42) -> int:
    root, output = Path(data_dir), Path(output_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}. Supply a GTZAN directory containing genre folders.")
    features: list[np.ndarray] = []; labels: list[int] = []; metadata: list[dict] = []
    for genre, label in LABEL_TO_ID.items():
        genre_dir = root / genre
        if not genre_dir.is_dir():
            continue
        for file in sorted(p for p in genre_dir.rglob("*") if p.suffix.lower() in {".wav", ".au", ".mp3", ".ogg", ".flac"}):
            track_id = f"{genre}/{file.stem}"
            for index, segment in enumerate(segment_audio(load_audio(file))):
                features.append(pool_to_features(segment)); labels.append(label)
                metadata.append({"track_id": track_id, "segment_index": index, "genre": genre, "label": label, "source": str(file)})
    if not features:
        raise ValueError("No supported audio files were found under the four target genre directories.")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "features.npy", np.stack(features)); np.save(output / "labels.npy", np.array(labels, dtype=np.int64))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    save_splits(track_level_split(metadata, seed=seed), splits_path)
    return len(features)
