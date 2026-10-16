"""Reusable loss functions."""
import torch
from torch import nn

def hybrid_loss(reconstruction: torch.Tensor, logits: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor, gamma: float = 0.9) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 <= gamma <= 1: raise ValueError("gamma must be between zero and one")
    reconstruction_loss = nn.functional.mse_loss(reconstruction, inputs)
    classification_loss = nn.functional.cross_entropy(logits, targets)
    return gamma * reconstruction_loss + (1 - gamma) * classification_loss, reconstruction_loss, classification_loss

def source_joint_loss(reconstruction: torch.Tensor, logits: torch.Tensor, inputs: torch.Tensor,
                      targets: torch.Tensor, classification_weight: float = 0.1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Original repository objective: MSE + 0.1 * cross-entropy."""
    reconstruction_loss = nn.functional.mse_loss(reconstruction, inputs)
    classification_loss = nn.functional.cross_entropy(logits, targets)
    return reconstruction_loss + classification_weight * classification_loss, reconstruction_loss, classification_loss
