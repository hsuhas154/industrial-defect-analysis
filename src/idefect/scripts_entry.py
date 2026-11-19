"""Console-script entry points declared in pyproject.toml.

These thin wrappers let the tool be driven either via ``python scripts/*.py`` or
via the installed console commands (``idefect-train-cnn`` etc.).
"""

from __future__ import annotations

import argparse

from idefect.config import load_config

# NOTE: torch-dependent imports (the CNN trainer, the Predictor's CNN backend)
# are done lazily inside the functions below so the classical branch and the
# synthetic-data generator work even if PyTorch is not installed.


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default="configs/neu_baseline.yaml",
        help="Path to the YAML config (default: configs/neu_baseline.yaml)",
    )


def gen_dummy_main() -> None:
    from idefect.data.synthetic import generate_dummy_dataset

    p = argparse.ArgumentParser(description="Generate a synthetic NEU-style dataset.")
    p.add_argument("--out", default="data/raw/NEU-CLS", help="Output directory")
    p.add_argument("--per-class", type=int, default=40, help="Images per class")
    p.add_argument("--size", type=int, default=200, help="Image size (px)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = generate_dummy_dataset(args.out, args.per_class, args.size, args.seed)
    print(f"Wrote synthetic dataset to {out}")


def train_classical_main() -> None:
    from idefect.training.train_classical_ml import train_classical_ml

    p = argparse.ArgumentParser(description="Train HOG + Decision Tree / SVM baselines.")
    _add_config_arg(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    train_classical_ml(cfg, config_path=args.config)


def train_cnn_main() -> None:
    from idefect.training.train_cnn import train_cnn

    p = argparse.ArgumentParser(description="Train the custom CNN baseline.")
    _add_config_arg(p)
    args = p.parse_args()
    cfg = load_config(args.config)
    train_cnn(cfg, config_path=args.config)


def predict_main() -> None:
    from idefect.inference.predict import Predictor

    p = argparse.ArgumentParser(description="Predict the defect class of one image.")
    p.add_argument("image", help="Path to the image to classify")
    p.add_argument(
        "--model",
        required=True,
        help="Path to a trained artifact (.joblib for classical, .pt for CNN)",
    )
    args = p.parse_args()
    predictor = Predictor(args.model)
    result = predictor.predict(args.image)
    print(f"model:      {result.model_kind}")
    print(f"prediction: {result.label}")
    if result.confidence is not None:
        print(f"confidence: {result.confidence:.4f}")
    if result.probabilities:
        print("probabilities:")
        for cls, prob in sorted(result.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"  {cls:<18} {prob:.4f}")


def dataset_report_main() -> None:
    import json
    from pathlib import Path

    from idefect.data.datasets import build_split
    from idefect.data.stats import audit_samples, audit_split, problems
    from idefect.visualization import eda

    p = argparse.ArgumentParser(
        description="Dataset statistics, sanity checks and exploratory figures."
    )
    _add_config_arg(p)
    args = p.parse_args()
    cfg = load_config(args.config)

    samples, class_names, split, _ = build_split(cfg)
    print(f"[data] {len(samples)} images, {len(class_names)} classes under {cfg.data.root}")
    report = audit_samples(samples, class_names)
    split_report = audit_split(split, report["duplicate_groups"])
    issues = problems(report, split_report)
    notes = []
    if report["n_duplicate_images"] and not split_report["duplicate_groups_across_splits"]:
        notes.append(f"{report['n_duplicate_images']} exact duplicate image(s) "
                     f"{report['duplicate_groups']} kept inside one split (no leakage)")

    out = Path(cfg.output.results_dir) / "dataset_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"dataset": report, "split": split_report, "issues": issues, "notes": notes},
                  fh, indent=2)

    fig_dir = Path(cfg.output.figures_dir) / "eda"
    eda.plot_class_balance(report["per_class_counts"], fig_dir / "class_balance.png")
    eda.plot_samples_by_class(samples, class_names, fig_dir / "samples_by_class.png",
                              seed=cfg.seed)
    eda.plot_preprocessing_comparison(samples, class_names,
                                      fig_dir / "preprocessing_comparison.png",
                                      image_size=cfg.data.image_size, seed=cfg.seed)
    eda.plot_intensity_histograms(samples, class_names, fig_dir / "intensity_histograms.png")
    eda.plot_class_mean_images(samples, class_names, fig_dir / "class_mean_images.png",
                               image_size=cfg.data.image_size)

    print(f"[data] counts: {report['per_class_counts']}")
    print(f"[data] shapes: {report['image_shapes']}  dtypes: {report['dtypes']}")
    print(f"[split] {split_report['sizes']}  disjoint={split_report['is_disjoint']}")
    for n in notes:
        print(f"[note] {n}")
    if issues:
        print("[check] ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("[check] all sanity checks passed")
    print(f"[out] {out} and figures in {fig_dir}")


def error_analysis_main() -> None:
    import json
    from pathlib import Path

    from idefect.evaluation.error_analysis import analyze_predictions

    p = argparse.ArgumentParser(description="Error analysis from saved test predictions.")
    _add_config_arg(p)
    p.add_argument("--predictions", nargs="*", default=None,
                   help="Specific *_test_predictions.csv files (default: all in results_dir)")
    args = p.parse_args()
    cfg = load_config(args.config)
    results_dir = Path(cfg.output.results_dir)
    files = [Path(f) for f in args.predictions] if args.predictions else sorted(
        results_dir.glob("*_test_predictions.csv"))
    if not files:
        raise SystemExit(f"no *_test_predictions.csv in {results_dir}; train a model first")

    reports = {}
    for f in files:
        r = analyze_predictions(f, Path(cfg.output.figures_dir) / "errors")
        reports[r["model"]] = r
        top = ", ".join(f"{c['true']}->{c['pred']} x{c['count']}" for c in r["confused_pairs"][:3])
        print(f"[errors:{r['model']}] {r['n_wrong']}/{r['n_test']} wrong; top confusions: "
              f"{top or 'none'}")
    out = results_dir / "error_analysis.json"
    with open(out, "w") as fh:
        json.dump(reports, fh, indent=2)
    print(f"[out] {out}")


def compare_main() -> None:
    from pathlib import Path

    from idefect.evaluation.compare import collect_metrics, markdown_table, update_readme, \
        write_summary

    p = argparse.ArgumentParser(description="Build the model comparison table and chart.")
    _add_config_arg(p)
    p.add_argument("--update-readme", action="store_true",
                   help="Rewrite the results block between the RESULTS markers in README.md")
    p.add_argument("--readme", default="README.md")
    args = p.parse_args()
    cfg = load_config(args.config)
    paths = write_summary(cfg.output.results_dir, cfg.output.figures_dir)
    print(Path(paths["markdown"]).read_text())
    for k, v in paths.items():
        print(f"[out] {k}: {v}")
    if args.update_readme:
        changed = update_readme(args.readme, markdown_table(collect_metrics(cfg.output.results_dir)))
        print(f"[readme] {'updated' if changed else 'already up to date'}")
