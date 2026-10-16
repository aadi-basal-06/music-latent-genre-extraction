import torch
from music_genre.models import TwoLayerClassifier, VanillaAutoencoder, DeepSoftmaxAutoencoder

def test_model_shapes():
    x=torch.randn(3,500)
    assert TwoLayerClassifier()(x).shape == (3,4)
    assert VanillaAutoencoder()(x).shape == (3,500)
    reconstruction, logits, embedding=DeepSoftmaxAutoencoder()(x)
    assert reconstruction.shape == (3,500) and logits.shape == (3,4) and embedding.shape == (3,64)
