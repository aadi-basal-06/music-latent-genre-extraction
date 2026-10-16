import numpy as np
from music_genre.data.preprocessing import pool_to_features
from music_genre.data.segmentation import segment_audio

def test_pooling_is_exact_and_deterministic():
    source=np.arange(22050,dtype=np.float32); assert pool_to_features(source).shape == (500,); assert np.array_equal(pool_to_features(source),pool_to_features(source))
def test_pooling_is_contiguous_bin_average():
    source=np.arange(10,dtype=np.float32)
    assert np.allclose(pool_to_features(source, 3), [1.5, 5., 8.])
def test_segmentation_discards_tail(): assert len(segment_audio(np.zeros(22050*2+9))) == 2
