"""Vanilla raw-audio autoencoder."""
import torch
from torch import nn

class VanillaAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 500, latent_dim: int = 64):
        super().__init__()
        # Activation sequence follows the original TensorFlow implementation.
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.Sigmoid(), nn.Linear(256, 192), nn.Tanh(), nn.Dropout(.2), nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.Sigmoid(), nn.Linear(128, 192), nn.Sigmoid(), nn.Dropout(.2), nn.Linear(192, 256), nn.ReLU(), nn.Linear(256, input_dim), nn.ReLU())
    def encode(self, x: torch.Tensor) -> torch.Tensor: return self.encoder(x)
    def decode(self, z: torch.Tensor) -> torch.Tensor: return self.decoder(z)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.decode(self.encode(x))
