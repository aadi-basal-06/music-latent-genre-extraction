"""Audio loading with explicit mono and sampling-rate behavior."""
from pathlib import Path
import librosa
import numpy as np

def load_audio(path: str | Path, sample_rate: int = 22050) -> np.ndarray:
    """Return mono float32 audio, resampled by librosa to ``sample_rate``."""
    samples, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(samples, dtype=np.float32)
