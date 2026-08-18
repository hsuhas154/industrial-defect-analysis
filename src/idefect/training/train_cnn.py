"""Train the custom CNN branch: preprocessing -> CNN -> classification.

Trains on the same deterministic split used by the classical branch, tracks
train/val loss and accuracy per epoch, keeps the best-validation checkpoint, and
evaluates that checkpoint on the held-out test split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from idefect.config import Config
from idefect.data.datasets import NEUImageDataset, load_samples, make_splits
from idefect.evaluation.classification import (
    compute_metrics,
    format_report,
    save_metrics,
    summarize,
)
from idefect.models.custom_cnn import CustomCNN, count_parameters
from idefect.seed import set_seed
from idefect.visualization.plots import plot_confusion_matrix, plot_training_curves


def resolve_device(pref: str) -> torch.device:
    """Resolve the 'auto'|'cpu'|'cuda' preference to a torch device."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            print("[warn] cuda requested but not available; falling back to cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    torch.set_grad_enabled(True)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds: List[int] = []
    trues: List[int] = []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(labels.tolist())
    return np.array(trues), np.array(preds)


def train_cnn(cfg: Config) -> Dict:
    """Train the custom CNN and evaluate on the test split."""
    set_seed(cfg.seed)
    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)
    device = resolve_device(cfg.cnn.device)
    print(f"[cnn] device = {device}")

    samples, class_names = load_samples(cfg.data.root)
    split = make_splits(
        samples,
        class_names,
        cfg.data.train_frac,
        cfg.data.val_frac,
        cfg.data.test_frac,
        cfg.seed,
    )
    print(f"[data] {split.summary()}")

    train_ds = NEUImageDataset(
        split.train, cfg.preprocessing, cfg.data.image_size,
        augment=cfg.cnn.augment, seed=cfg.seed,
    )
    val_ds = NEUImageDataset(
        split.val, cfg.preprocessing, cfg.data.image_size, augment=False, seed=cfg.seed,
    )
    test_ds = NEUImageDataset(
        split.test, cfg.preprocessing, cfg.data.image_size, augment=False, seed=cfg.seed,
    )

    common = dict(batch_size=cfg.cnn.batch_size, num_workers=cfg.cnn.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = CustomCNN(
        num_classes=split.num_classes,
        in_channels=cfg.cnn.in_channels,
        base_channels=cfg.cnn.base_channels,
        dropout=cfg.cnn.dropout,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[cnn] CustomCNN with {n_params:,} trainable parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.cnn.lr, weight_decay=cfg.cnn.weight_decay
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    ckpt_path = results_dir / "custom_cnn.pt"

    t0 = time.time()
    for epoch in range(1, cfg.cnn.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        print(
            f"[epoch {epoch:2d}/{cfg.cnn.epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    print(f"[cnn] training finished in {train_time:.1f}s; best val acc={best_val_acc:.4f}")

    # Restore best and save a self-contained checkpoint bundle.
    if best_state is not None:
        model.load_state_dict(best_state)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": split.num_classes,
            "image_size": cfg.data.image_size,
            "cnn_config": cfg.cnn.__dict__,
            "preprocessing": cfg.preprocessing,
            "n_params": n_params,
        },
        ckpt_path,
    )

    # Final evaluation on the held-out test split.
    y_true, y_pred = _predict(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["best_val_acc"] = round(best_val_acc, 4)
    metrics["n_params"] = n_params
    print(f"[eval:cnn] {summarize(metrics)}")
    print(format_report(y_true, y_pred, class_names))

    save_metrics(metrics, results_dir / "custom_cnn_metrics.json")
    with open(results_dir / "custom_cnn_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        figures_dir / "custom_cnn_confusion_matrix.png",
        title="Custom CNN — confusion matrix",
    )
    plot_training_curves(history, figures_dir / "custom_cnn_training_curves.png")

    return metrics
