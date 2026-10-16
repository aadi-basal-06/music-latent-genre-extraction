"""PCA embedding visualization."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from music_genre import ID_TO_LABEL

def plot_pca(embeddings: np.ndarray, labels: np.ndarray, path: str | Path, title: str, seed: int = 42) -> np.ndarray:
    reduced=PCA(n_components=2, random_state=seed).fit_transform(embeddings); path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax=plt.subplots(figsize=(8, 6))
    for label, name in ID_TO_LABEL.items():
        points=reduced[labels == label]; ax.scatter(points[:,0], points[:,1], s=14, alpha=.7, label=name)
    ax.set(title=title, xlabel="Principal component 1", ylabel="Principal component 2"); ax.legend(title="True genre"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return reduced
