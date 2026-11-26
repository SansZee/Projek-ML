"""Loss function, class weighting anti-imbalance, metrik evaluasi, dan ekspor audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


class FocalLoss(nn.Module):
    """Multi-class Focal Loss opsional dengan class weights dari set train."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, targets, weight=self.weight, reduction="none")
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Hitung inverse-frequency class weights yang distabilkan dengan akar kuadrat."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    raw_weights = counts.sum() / (num_classes * counts)
    # Akar kuadrat mencegah gradien kelas sangat minoritas (mis. jijik) meledak berlebihan.
    balanced = np.sqrt(raw_weights)
    balanced = balanced / balanced.mean()
    return torch.tensor(balanced, dtype=torch.float32)


def build_loss_fn(
    name: str,
    train_labels: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """Bangun loss standar sesuai ruang pencarian Optuna di ``rencana-pelatihan.md``."""
    weights = compute_class_weights(train_labels, num_classes).to(device)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "label_smooth_0.1":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if name == "focal_gamma_2":
        return FocalLoss(weight=weights, gamma=2.0).to(device)
    raise ValueError(f"loss_fn tidak dikenal: {name}")


def evaluate_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    classes: tuple[str, ...],
) -> dict[str, object]:
    """Hitung Macro-F1, Accuracy, F1 per kelas, dan Confusion Matrix."""
    probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()
    preds = probs.argmax(axis=-1)
    labels_idx = list(range(len(classes)))

    macro_f1 = float(f1_score(targets, preds, labels=labels_idx, average="macro", zero_division=0))
    acc = float(accuracy_score(targets, preds))
    per_class = f1_score(targets, preds, labels=labels_idx, average=None, zero_division=0)
    cm = confusion_matrix(targets, preds, labels=labels_idx)

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "per_class_f1": {cls: float(score) for cls, score in zip(classes, per_class, strict=True)},
        "confusion_matrix": cm.tolist(),
        "probs": probs,
        "preds": preds,
    }


def export_audit_predictions(
    utterance_ids: list[str],
    probs: np.ndarray,
    classes: tuple[str, ...],
    out_csv: Path,
) -> Path:
    """Tulis CSV format ``utterance_id,predicted,score`` untuk ``scripts/12_audit.py``."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_idx = probs.argmax(axis=-1)
    scores = probs.max(axis=-1)
    frame = pd.DataFrame(
        {
            "utterance_id": utterance_ids,
            "predicted": [classes[i] for i in pred_idx],
            "score": np.round(scores, 6),
        }
    )
    frame.to_csv(out_csv, index=False)
    return out_csv
