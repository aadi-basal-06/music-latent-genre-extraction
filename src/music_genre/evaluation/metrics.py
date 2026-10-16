"""Classification metrics and confusion-matrix persistence."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from music_genre import ID_TO_LABEL

def classification_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict:
    labels = list(range(4)); names = [ID_TO_LABEL[x] for x in labels]
    return {"accuracy": float(accuracy_score(targets, predictions)), "report": classification_report(targets, predictions, labels=labels, target_names=names, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist()}

def save_metrics(metrics: dict, path: str | Path) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(metrics, indent=2))

def plot_confusion_matrix(metrics: dict, path: str | Path) -> None:
    """Save a labelled confusion matrix for a classification evaluation."""
    matrix = np.asarray(metrics["confusion_matrix"]); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    names = [ID_TO_LABEL[i] for i in range(4)]; ax.set(xticks=range(4), yticks=range(4), xticklabels=names, yticklabels=names, xlabel="Predicted genre", ylabel="True genre", title="Confusion matrix")
    for row in range(4):
        for column in range(4): ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
