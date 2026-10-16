"""Generic, compact training loop for all three objectives."""
from pathlib import Path
import csv
import torch
from torch.utils.data import DataLoader
from music_genre.models.losses import hybrid_loss, source_joint_loss
from music_genre.utils.checkpoint import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, device, config, mode: str, gamma: float = 0.9,
                 loss_mode: str = "convex", classification_weight: float = 0.1):
        self.model, self.optimizer, self.device, self.config, self.mode, self.gamma = model.to(device), optimizer, device, config, mode, gamma
        self.loss_mode, self.classification_weight = loss_mode, classification_weight
    def _run(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train); sums = {"loss": 0., "reconstruction_loss": 0., "classification_loss": 0., "correct": 0.}; count = 0
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device); self.optimizer.zero_grad()
                output = self.model(x)
                if self.mode == "baseline":
                    logits = output; loss = torch.nn.functional.cross_entropy(logits, y); rec = torch.tensor(0., device=self.device); cls = loss
                elif self.mode == "autoencoder":
                    rec_out = output; loss = torch.nn.functional.mse_loss(rec_out, x); rec = loss; cls = torch.tensor(0., device=self.device); logits = None
                else:
                    rec_out, logits, _ = output
                    if self.loss_mode == "source_additive":
                        loss, rec, cls = source_joint_loss(rec_out, logits, x, y, self.classification_weight)
                    else:
                        loss, rec, cls = hybrid_loss(rec_out, logits, x, y, self.gamma)
                if train: loss.backward(); self.optimizer.step()
                n = len(x); count += n; sums["loss"] += loss.item() * n; sums["reconstruction_loss"] += rec.item() * n; sums["classification_loss"] += cls.item() * n
                if logits is not None: sums["correct"] += (logits.argmax(1) == y).sum().item()
        return {"loss": sums["loss"] / count, "reconstruction_loss": sums["reconstruction_loss"] / count, "classification_loss": sums["classification_loss"] / count, "accuracy": sums["correct"] / count if self.mode != "autoencoder" else 0.}
    def fit(self, train_loader, val_loader, epochs: int, checkpoint_dir: str | Path, output_dir: str | Path, seed: int, start_epoch: int = 0):
        best = float("inf") if self.mode == "autoencoder" else -float("inf"); history=[]; checkpoint_dir=Path(checkpoint_dir)
        for epoch in range(start_epoch, start_epoch + epochs):
            train = self._run(train_loader, True); val = self._run(val_loader, False); row={"epoch": epoch + 1, **{f"train_{k}":v for k,v in train.items()}, **{f"val_{k}":v for k,v in val.items()}}; history.append(row)
            metric = -val["loss"] if self.mode == "autoencoder" else val["accuracy"]
            if metric > best: best=metric; save_checkpoint(checkpoint_dir / "best.pt", self.model, self.optimizer, epoch + 1, best, self.config, seed)
            save_checkpoint(checkpoint_dir / "last.pt", self.model, self.optimizer, epoch + 1, best, self.config, seed)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with (Path(output_dir) / "history.csv").open("w", newline="") as handle: csv.DictWriter(handle, fieldnames=history[0].keys()).writeheader(); handle.seek(0, 2); csv.DictWriter(handle, fieldnames=history[0].keys()).writerows(history)
        return history
