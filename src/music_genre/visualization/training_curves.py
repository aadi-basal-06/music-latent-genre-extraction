"""Training curve plotting."""
from pathlib import Path
import matplotlib.pyplot as plt

def plot_training_curves(history: list[dict], path: str | Path) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig, ax=plt.subplots(figsize=(8,5)); ax.plot([x['epoch'] for x in history], [x['train_loss'] for x in history], label='train loss'); ax.plot([x['epoch'] for x in history], [x['val_loss'] for x in history], label='validation loss'); ax.set(xlabel='Epoch', ylabel='Loss', title='Training curve'); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
