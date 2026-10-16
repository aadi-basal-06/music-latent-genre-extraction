"""Fixed-duration waveform segmentation."""
import numpy as np

def segment_audio(samples: np.ndarray, sample_rate: int = 22050, seconds: float = 1.0) -> list[np.ndarray]:
    """Split into complete fixed-size segments; discard an incomplete tail."""
    length = int(sample_rate * seconds)
    if length <= 0:
        raise ValueError("segment duration must be positive")
    return [samples[i:i + length] for i in range(0, len(samples) - length + 1, length)]
