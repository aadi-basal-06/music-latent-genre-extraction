"""Paper-specified 500 -> 128 -> 4 baseline."""
import torch
from torch import nn

class TwoLayerClassifier(nn.Module):
    def __init__(self, input_dim: int = 500, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.network(x)
