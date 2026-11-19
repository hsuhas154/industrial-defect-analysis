"""Train the custom CNN branch: preprocessing -> CNN -> classification.

Trains on the same deterministic split used by the classical branch, tracks
train/val loss, accuracy and macro-F1 per epoch, keeps the best checkpoint by a
configurable validation metric, and evaluates that checkpoint once on the
held-out test split.

A note on BatchNorm: the stored running mean/variance only approach the true
data statistics after enough optimizer steps. On very small datasets (a few
batches per epoch) early-epoch validation scores can therefore look far worse
than training scores even though the network has learned the task. This is
why the checkpoint is selected on validation performance across the whole run
rather than taken from a fixed epoch.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from idefect.config import Config
from idefect.data.datasets import NEUImageDataset, build_split, seed_worker
from idefect.evaluation.classification import (
    compute_metrics,
    format_report,
    save_metrics,
    save_predictions,
    summarize,
)
from idefect.experiment_log import config_hash, environment_info, log_run
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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_epoch(model, loader, criterion, device, optimizer) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None):
    """Run ``model`` over ``loader`` in eval mode.

    Returns ``(y_true, y_pred, probabilities, mean_loss)``; ``mean_loss`` is
    ``None`` when no criterion is given.
    """
    model.eval()
    trues: List[int] = []
    probs: List[np.ndarray] = []
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, labels.to(device)).item() * images.size(0)
            n += images.size(0)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        trues.extend(labels.tolist())
    p = np.concatenate(probs) if probs else np.zeros((0, 0))
    y_true = np.asarray(trues)
    y_pred = p.argmax(1) if len(p) else np.asarray([], dtype=int)
    mean_loss = total_loss / n if criterion is not None and n else None
    return y_true, y_pred, p, mean_loss


def _is_better(metric: str, current: Dict[str, float], best: Dict[str, float]) -> bool:
    """Compare epoch summaries; ties on the main metric fall back to val loss."""
    if not best:
        return True
    if metric == "loss":
        return current["val_loss"] < best["val_loss"]
    key = "val_macro_f1" if metric == "macro_f1" else "val_acc"
    if current[key] != best[key]:
        return current[key] > best[key]
    return current["val_loss"] < best["val_loss"]


def _make_loaders(cfg: Config, split, device: torch.device):
    cache = cfg.cnn.cache_images
    train_ds = NEUImageDataset(
        split.train, cfg.preprocessing, cfg.data.image_size,
        augment=cfg.cnn.augment, cache=cache,
    )
    val_ds = NEUImageDataset(split.val, cfg.preprocessing, cfg.data.image_size, cache=cache)
    test_ds = NEUImageDataset(split.test, cfg.preprocessing, cfg.data.image_size, cache=cache)

    workers = cfg.cnn.num_workers
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    common = dict(
        batch_size=cfg.cnn.batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def _make_scheduler(cfg: Config, optimizer):
    if cfg.cnn.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cnn.epochs)
    if cfg.cnn.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
    return None


def train_cnn(cfg: Config, config_path: str = "") -> Dict:
    """Train the custom CNN and evaluate the selected checkpoint on the test split."""
    set_seed(cfg.seed)
    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)
    device = resolve_device(cfg.cnn.device)
    print(f"[cnn] device = {device}")

    _, class_names, split, dup_groups = build_split(cfg)
    if dup_groups:
        print(f"[data] {len(dup_groups)} exact-duplicate group(s) kept within one split")
    print(f"[data] {split.summary()}")

    train_loader, val_loader, test_loader = _make_loaders(cfg, split, device)

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
    scheduler = _make_scheduler(cfg, optimizer)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "lr": [],
    }
    best: Dict[str, float] = {}
    best_state = None
    stale = 0
    patience = cfg.cnn.early_stopping_patience

    t0 = time.time()
    for epoch in range(1, cfg.cnn.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, device, optimizer)
        yv, pv, _, va_loss = predict_loader(model, val_loader, device, criterion)
        va_acc = float((yv == pv).mean()) if len(yv) else 0.0
        va_f1 = float(f1_score(yv, pv, average="macro", zero_division=0)) if len(yv) else 0.0

        for k, v in (("train_loss", tr_loss), ("train_acc", tr_acc), ("val_loss", va_loss),
                     ("val_acc", va_acc), ("val_macro_f1", va_f1), ("lr", lr)):
            history[k].append(v)
        print(
            f"[epoch {epoch:3d}/{cfg.cnn.epochs}] lr={lr:.2e} "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}"
        )

        current = {"val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                   "epoch": epoch}
        if _is_better(cfg.cnn.select_metric, current, best):
            best = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(va_loss)
        elif scheduler is not None:
            scheduler.step()

        if patience and stale >= patience:
            print(f"[cnn] early stopping at epoch {epoch} (no improvement for {patience})")
            break

    train_time = time.time() - t0
    print(
        f"[cnn] training finished in {train_time:.1f}s; best epoch={best.get('epoch')} "
        f"val_acc={best.get('val_acc', 0):.4f} val_f1={best.get('val_macro_f1', 0):.4f}"
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": split.num_classes,
            "image_size": cfg.data.image_size,
            "cnn_config": dict(cfg.cnn.__dict__),
            "preprocessing": cfg.preprocessing,
            "n_params": n_params,
            "best_epoch": best.get("epoch"),
            "config_hash": config_hash(cfg),
        },
        results_dir / "custom_cnn.pt",
    )

    # Final, single evaluation on the held-out test split.
    t1 = time.time()
    y_true, y_pred, y_proba, _ = predict_loader(model, test_loader, device)
    infer_time = time.time() - t1
    metrics = compute_metrics(y_true, y_pred, class_names, y_proba=y_proba)
    metrics.update(
        model="custom_cnn",
        train_time_sec=round(train_time, 2),
        test_inference_sec=round(infer_time, 3),
        epochs_run=len(history["train_loss"]),
        best_epoch=best.get("epoch"),
        best_val_acc=round(best.get("val_acc", 0.0), 4),
        best_val_macro_f1=round(best.get("val_macro_f1", 0.0), 4),
        n_params=n_params,
        device=str(device),
        config_hash=config_hash(cfg),
        environment=environment_info(),
    )
    print(f"[eval:cnn] {summarize(metrics)}")
    print(format_report(y_true, y_pred, class_names))

    save_metrics(metrics, results_dir / "custom_cnn_metrics.json")
    save_predictions(
        results_dir / "custom_cnn_test_predictions.csv",
        [p for p, _ in split.test], y_true, y_pred, class_names, y_proba,
    )
    with open(results_dir / "custom_cnn_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        figures_dir / "custom_cnn_confusion_matrix.png",
        title="Custom CNN confusion matrix",
    )
    plot_training_curves(history, figures_dir / "custom_cnn_training_curves.png")
    log_run(cfg.output.logs_dir, "custom_cnn", cfg, metrics, config_path=config_path)
    return metrics
