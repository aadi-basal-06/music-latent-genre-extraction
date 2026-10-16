"""Shared-encoder reconstruction and genre classifier model."""
import torch
from torch import nn

class DeepSoftmaxAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 500, latent_dim: int = 64, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        # Source verification: sigmoid -> tanh -> dropout -> ReLU encoder;
        # sigmoid -> sigmoid -> dropout -> ReLU decoder; tanh classifier.
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.Sigmoid(), nn.Linear(256, 192), nn.Tanh(), nn.Dropout(dropout), nn.Linear(192, 128), nn.ReLU(), nn.Linear(128, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.Sigmoid(), nn.Linear(128, 192), nn.Sigmoid(), nn.Dropout(dropout), nn.Linear(192, 256), nn.ReLU(), nn.Linear(256, input_dim), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, num_classes))
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        return self.decoder(embedding), self.classifier(embedding), embedding
