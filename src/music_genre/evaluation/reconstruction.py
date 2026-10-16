"""Reconstruction-error evaluation."""
import torch

def reconstruction_mse(model, loader, device) -> float:
    model.eval(); total=0.; count=0
    with torch.no_grad():
        for x, _ in loader:
            x=x.to(device); output=model(x); reconstruction = output[0] if isinstance(output, tuple) else output
            total += torch.nn.functional.mse_loss(reconstruction, x, reduction="sum").item(); count += x.numel()
    return total / count
