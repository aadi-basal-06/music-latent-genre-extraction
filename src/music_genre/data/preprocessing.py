"""Deterministic pooling of raw waveform segments."""
import numpy as np

def pool_to_features(segment: np.ndarray, dimensions: int = 500) -> np.ndarray:
    """Resample by deterministic equal-width bin averaging to exactly ``dimensions``.

    ``numpy.array_split`` divides the source into 500 contiguous bins differing
    by at most one sample; each feature is that bin's arithmetic mean. This
    uses all 22,050 samples at 22,050 Hz and avoids assuming the paper's
    inconsistent pooling factor.
    """
    values = np.asarray(segment, dtype=np.float32).reshape(-1)
    if values.size == 0 or dimensions <= 0:
        raise ValueError("segment and dimensions must be non-empty/positive")
    return np.asarray([bin_.mean() for bin_ in np.array_split(values, dimensions)], dtype=np.float32)
