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
from typing import Dict, List, Optional, Sequence, Tuple

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
from idefect.inference.input_check import confidence_floor, reference_stats
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
        # forkserver avoids fork() from a multi-threaded process (CUDA and
        # BLAS threads are already running), which can deadlock workers.
        multiprocessing_context="forkserver" if workers > 0 else None,
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


def _train_once(cfg: Config, seed: int, out_dir: Path, fig_dir: Path, config_path: str = "",
                verbose: bool = True, split_override=None, evaluate_test: bool = True) -> Dict:
    """Train one CNN with ``seed`` on the shared split; write its artifacts to ``out_dir``.

    With ``evaluate_test=False`` (used by tuning studies such as the epoch
    study) the test split is never touched: only training and validation
    numbers are produced.
    """
    set_seed(seed)
    device = resolve_device(cfg.cnn.device)
    if split_override is not None:  # used by the leakage study only
        split, groups, class_names = split_override, [], split_override.class_names
    else:
        _, class_names, split, groups = build_split(cfg)
    if verbose:
        print(f"[cnn] seed={seed} device={device} | {split.summary()} | "
              f"{len(groups)} near-duplicate group(s) kept within one split")

    run_cfg = Config(**{**cfg.__dict__})
    run_cfg.seed = seed
    train_loader, val_loader, test_loader = _make_loaders(run_cfg, split, device)

    model = CustomCNN(
        num_classes=split.num_classes,
        in_channels=cfg.cnn.in_channels,
        base_channels=cfg.cnn.base_channels,
        dropout=cfg.cnn.dropout,
    ).to(device)
    n_params = count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.cnn.lr, weight_decay=cfg.cnn.weight_decay)
    scheduler = _make_scheduler(cfg, optimizer)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": [],
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
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == cfg.cnn.epochs):
            print(f"[epoch {epoch:3d}/{cfg.cnn.epochs}] lr={lr:.2e} train_loss={tr_loss:.4f} "
                  f"train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}")
        current = {"val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1, "epoch": epoch}
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
            if verbose:
                print(f"[cnn] early stopping at epoch {epoch}")
            break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    yv_true, _, pv_val, _ = predict_loader(model, val_loader, device)
    reference = {"image": reference_stats([p for p, _ in split.train]),
                 "confidence_floor": confidence_floor(pv_val, yv_true)}
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
            "seed": seed,
            "config_hash": config_hash(cfg),
            "reference": reference,
        },
        out_dir / "custom_cnn.pt",
    )
    with open(out_dir / "custom_cnn_history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    if not evaluate_test:
        return {
            "seed": seed,
            "epochs_run": len(history["train_loss"]),
            "best_epoch": best.get("epoch"),
            "best_val_acc": round(best.get("val_acc", 0.0), 4),
            "best_val_macro_f1": round(best.get("val_macro_f1", 0.0), 4),
            "train_time_sec": round(train_time, 2),
            "device": str(device),
        }
    t1 = time.time()
    y_true, y_pred, y_proba, _ = predict_loader(model, test_loader, device)
    infer_time = time.time() - t1
    metrics = compute_metrics(y_true, y_pred, class_names, y_proba=y_proba)
    metrics.update(
        model="custom_cnn",
        display_name="Custom CNN",
        branch="cnn",
        seed=seed,
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
    save_metrics(metrics, out_dir / "custom_cnn_metrics.json")
    save_predictions(out_dir / "custom_cnn_test_predictions.csv", [p for p, _ in split.test],
                     y_true, y_pred, class_names, y_proba)
    log_run(cfg.output.logs_dir, f"custom_cnn_seed{seed}", cfg, metrics, config_path=config_path)
    if verbose:
        print(f"[eval:cnn seed={seed}] {summarize(metrics)} | best epoch {best.get('epoch')} "
              f"val_f1 {best.get('val_macro_f1', 0):.4f} | {train_time:.1f}s")
    return metrics


def train_cnn(cfg: Config, config_path: str = "", seeds: Optional[Sequence[int]] = None) -> Dict:
    """Train the custom CNN once per seed on the same split.

    The artifact kept as ``custom_cnn.pt`` is the seed with the best
    *validation* macro-F1 (never chosen on test). Test metrics are reported for
    that model and as mean and standard deviation over all seeds.
    """
    import shutil

    seeds = list(seeds) if seeds else list(cfg.cnn.seeds) or [cfg.seed]
    results_dir = Path(cfg.output.results_dir)
    figures_dir = Path(cfg.output.figures_dir)
    runs = {}
    for s in seeds:
        runs[s] = _train_once(cfg, s, results_dir / "cnn_runs" / f"seed{s}", figures_dir,
                              config_path=config_path)

    chosen = max(seeds, key=lambda s: (runs[s]["best_val_macro_f1"], -s))
    src = results_dir / "cnn_runs" / f"seed{chosen}"
    for name in ("custom_cnn.pt", "custom_cnn_test_predictions.csv", "custom_cnn_history.json"):
        shutil.copyfile(src / name, results_dir / name)
    metrics = dict(runs[chosen])

    def stat(key):
        v = np.array([runs[s][key] for s in seeds], dtype=float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()), "max": float(v.max())}

    metrics["seed_summary"] = {
        "seeds": seeds,
        "chosen_seed": chosen,
        "selection": "best validation macro-F1",
        "test_macro_f1": stat("macro_f1"),
        "test_accuracy": stat("accuracy"),
        "val_macro_f1": stat("best_val_macro_f1"),
        "train_time_sec": stat("train_time_sec"),
        "per_seed": {str(s): {"test_macro_f1": runs[s]["macro_f1"], "test_accuracy": runs[s]["accuracy"],
                              "val_macro_f1": runs[s]["best_val_macro_f1"],
                              "best_epoch": runs[s]["best_epoch"]} for s in seeds},
    }
    save_metrics(metrics, results_dir / "custom_cnn_metrics.json")
    with open(src / "custom_cnn_history.json") as fh:
        history = json.load(fh)
    plot_confusion_matrix(metrics["confusion_matrix"], metrics["class_names"],
                          figures_dir / "custom_cnn_confusion_matrix.png",
                          title=f"Custom CNN confusion matrix (seed {chosen})")
    plot_training_curves(history, figures_dir / "custom_cnn_training_curves.png")
    ss = metrics["seed_summary"]["test_macro_f1"]
    print(f"[cnn] {len(seeds)} seeds: test macro-F1 {ss['mean']:.4f} +- {ss['std']:.4f} "
          f"(min {ss['min']:.4f}, max {ss['max']:.4f}); kept seed {chosen} (best val)")
    return metrics
