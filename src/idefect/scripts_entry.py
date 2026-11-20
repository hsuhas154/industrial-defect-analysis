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

    p = argparse.ArgumentParser(description="Train the custom CNN (one run per seed).")
    _add_config_arg(p)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override cnn.seeds from the config, e.g. --seeds 42 43")
    p.add_argument("--epochs", type=int, default=None, help="Override cnn.epochs")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg.cnn.epochs = args.epochs
    train_cnn(cfg, config_path=args.config, seeds=args.seeds)


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
    from dataclasses import asdict
    from pathlib import Path

    from idefect.data.datasets import build_split, duplicate_report
    from idefect.data.stats import audit_samples, audit_split, problems, verify_manifest
    from idefect.visualization import eda
    from idefect.visualization.duplicates import plot_group_sizes, plot_pairs

    p = argparse.ArgumentParser(
        description="Dataset statistics, sanity checks, near-duplicate audit and EDA figures."
    )
    _add_config_arg(p)
    args = p.parse_args()
    cfg = load_config(args.config)

    manifest = verify_manifest(cfg.data.root, cfg.data.manifest) if cfg.data.manifest else {"checked": False}
    if manifest.get("checked"):
        bad = len(manifest["missing"]) + len(manifest["modified"]) + len(manifest["unexpected"])
        print(f"[data] manifest check: {manifest['n_expected']} files expected, "
              f"{'all intact' if not bad else str(bad) + ' problem(s)'}")
    samples, class_names, split, groups = build_split(cfg, verbose=True)
    print(f"[data] {len(samples)} images, {len(class_names)} classes under {cfg.data.root}")
    report = audit_samples(samples, class_names)
    dup = duplicate_report(cfg, samples) if cfg.data.duplicates.enabled else None
    dsum = dup.summary() if dup else None
    split_report = audit_split(split, report["duplicate_groups"], groups)
    issues = problems(report, split_report, dsum, manifest)
    notes = []
    if dsum:
        notes.append(f"{dsum['n_images_in_groups']} images in {dsum['n_groups']} near-duplicate "
                     f"group(s) (largest {dsum['largest_group']}); every group kept inside one split")
        if dup.unverified:
            notes.append(f"{len(dup.unverified)} similar pair(s) could not be verified either way "
                         "(too little background texture); kept apart, listed for inspection")

    res_dir = Path(cfg.output.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    out = res_dir / "dataset_report.json"
    with open(out, "w") as fh:
        json.dump({"dataset": report, "manifest": manifest, "split": split_report, "near_duplicates": dsum,
                   "issues": issues, "notes": notes}, fh, indent=2)
    if dup:
        with open(res_dir / "near_duplicates.json", "w") as fh:
            json.dump({"summary": dsum, "groups": dup.groups,
                       "pairs": [asdict(x) for x in dup.pairs if x.kind != "distinct"],
                       "closest_distinct": [asdict(x) for x in
                                            sorted((x for x in dup.pairs if x.kind == "distinct"
                                                    and x.grain is not None),
                                                   key=lambda x: -x.grain)[:50]]},
                      fh, indent=2)
        ddir = Path(cfg.output.figures_dir) / "duplicates"
        plot_group_sizes(dup.groups, ddir / "group_sizes.png")
        same = [x for x in dup.pairs if x.kind == "same"]
        plot_pairs([x for x in same if x.evidence == "pixel-identical"][::max(1, len(same) // 12)],
                   ddir / "same_pixel_identical.png",
                   "Grouped: pixel-identical overlaps (sample)")
        plot_pairs(sorted((x for x in same if x.evidence == "grain"), key=lambda x: x.grain),
                   ddir / "same_grain_weakest.png",
                   "Grouped: re-processed copies, weakest evidence first")
        plot_pairs(dup.unresolved_review, ddir / "review_pairs.png",
                   "Awaiting review (grouped for safety until decided)", max_rows=40)
        plot_pairs(sorted((x for x in dup.pairs if x.kind == "distinct" and x.grain is not None),
                          key=lambda x: -x.grain), ddir / "distinct_closest.png",
                   "Kept apart: the most similar pairs judged distinct")
        plot_pairs(sorted(dup.unverified, key=lambda x: -x.overlap_px), ddir / "unverified.png",
                   "Kept apart, not verifiable (too little background texture)")
        # The pairs that matter most for leakage: judged distinct, closest to the
        # review band, and actually placed in different splits.
        where = {p: n for n in ("train", "val", "test") for p, _ in getattr(split, n)}
        gid = {p: k for k, g in enumerate(dup.groups) for p in g}
        boundary = [x for x in sorted((x for x in dup.pairs if x.kind == "distinct" and x.grain is not None),
                                      key=lambda x: -x.grain)
                    if x.grain >= 0.25 and where.get(x.a) != where.get(x.b)
                    and not (x.a in gid and gid.get(x.a) == gid.get(x.b))]
        plot_pairs(boundary, ddir / "distinct_across_splits.png",
                   "Judged distinct and placed in different splits (grain >= 0.25)", max_rows=20)
        notes.append(f"{len(boundary)} distinct pair(s) with grain >= 0.25 sit in different splits; "
                     "see figures/duplicates/distinct_across_splits.png")

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
    print(f"[split] per class: {split_report['class_distribution']}")
    if dsum:
        print(f"[dups] {dsum['pairs_by_decision']}  evidence for grouped pairs: "
              f"{dsum['same_pairs_by_evidence']}")
    for n in notes:
        print(f"[note] {n}")
    if issues:
        print("[check] ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("[check] all sanity checks passed")
    print(f"[out] {out} and figures in {Path(cfg.output.figures_dir)}")


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
        changed = update_readme(
            args.readme, markdown_table(collect_metrics(cfg.output.results_dir), include_time=False))
        print(f"[readme] {'updated' if changed else 'already up to date'}")


def ablations_main() -> None:
    from pathlib import Path

    from idefect.training.ablations import run_ablations

    p = argparse.ArgumentParser(description="Classical-branch ablations (preprocessing, resolution, HOG).")
    _add_config_arg(p)
    p.add_argument("--feature-sets", nargs="*", default=["hog", "lbp", "glcm", "lbp+glcm"])
    p.add_argument("--models", nargs="*", default=["svm", "logreg"])
    p.add_argument("--hog-models", nargs="*", default=["svm"])
    p.add_argument("--only", nargs="*", default=["preprocessing", "resolution", "hog"],
                   choices=["preprocessing", "resolution", "hog"],
                   help="ablations to run; the others keep their saved results")
    args = p.parse_args()
    cfg = load_config(args.config)
    run_ablations(cfg, args.feature_sets, args.models, args.hog_models,
                  Path(cfg.output.results_dir) / "classical", kinds=args.only)


def classical_report_main() -> None:
    from idefect.evaluation.classical_report import make_classical_figures

    p = argparse.ArgumentParser(description="All classical-branch figures from saved results.")
    _add_config_arg(p)
    args = p.parse_args()
    out = make_classical_figures(load_config(args.config))
    for k, v in out.items():
        print(f"[fig] {k}: {v}")


def epoch_study_main() -> None:
    from pathlib import Path

    from idefect.training.studies import cnn_epoch_study

    p = argparse.ArgumentParser(description="CNN training length chosen on validation macro-F1.")
    _add_config_arg(p)
    p.add_argument("--epochs", type=int, nargs="*", default=[40, 80, 150])
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    args = p.parse_args()
    cfg = load_config(args.config)
    for row in cnn_epoch_study(cfg, args.epochs, args.seeds, Path(cfg.output.results_dir) / "cnn_epoch_study"):
        print(f"[epochs={row['epochs']}] val macro-F1 {row['val_macro_f1_mean']:.4f} +- {row['val_macro_f1_std']:.4f}")


def leakage_study_main() -> None:
    from pathlib import Path

    from idefect.training.studies import leakage_study

    p = argparse.ArgumentParser(description="Random split versus group-aware split.")
    _add_config_arg(p)
    p.add_argument("--classical", nargs="*", default=["lbp+glcm:logreg", "lbp+glcm:svm", "hog:svm"],
                   help="feature_set:model pairs")
    p.add_argument("--cnn-seeds", type=int, nargs="*", default=[42, 43, 44])
    args = p.parse_args()
    cfg = load_config(args.config)
    pairs = [tuple(x.split(":")) for x in args.classical]
    leakage_study(cfg, pairs, args.cnn_seeds, Path(cfg.output.results_dir) / "leakage_study")
