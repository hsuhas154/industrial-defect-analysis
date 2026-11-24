#!/usr/bin/env python3
"""Streamlit inference UI for the Industrial Surface Defect Analysis project.

Upload a steel-surface image, pick a trained model (the custom CNN or one of
the classical models on HOG / LBP / GLCM features), and view the predicted
defect class with per-class confidences and warnings for unusual inputs.
Optionally run every trained model on the same image to compare them side by
side, or check whether the upload is a copy of a NEU image.

Run with:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import json
import os
import sys

# Make src/ importable when launched via `streamlit run` from the repo root.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Work from the repo root, so dataset paths are spelled exactly as the scripts
# spell them (relative); the cached near-duplicate result is keyed on them.
if Path.cwd() != _ROOT:
    os.chdir(_ROOT)

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from idefect.inference.input_check import check_input, is_colour  # noqa: E402
from idefect.inference.predict import Predictor  # noqa: E402

# Trained artifacts are read from here; IDEFECT_RESULTS_DIR overrides it.
RESULTS_DIR = Path(os.environ.get("IDEFECT_RESULTS_DIR", _ROOT / "experiments" / "results"))

st.set_page_config(page_title="Industrial Surface Defect Analysis", layout="wide")


@st.cache_resource(show_spinner=False)
def load_predictor(artifact_path: str) -> Predictor:
    return Predictor(artifact_path)


def find_artifacts():
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.joblib")) + sorted(RESULTS_DIR.glob("*.pt"))


def load_test_metrics(artifact: Path):
    """Test-split metrics saved next to the artifact by the training script."""
    path = artifact.with_name(f"{artifact.stem}_metrics.json")
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


@st.cache_resource(show_spinner="Loading the dataset index (once per session)...")
def dataset_index(config_path: str = "configs/neu_baseline.yaml"):
    """Dataset paths and the split each image belongs to (None if no data).

    Uses the cached near-duplicate result only: if it is missing, the cached
    call raises ``DuplicateCacheMissing`` (exceptions are not cached, so the
    next attempt checks again) instead of running a detection for minutes.
    """
    from idefect.config import load_config
    from idefect.data.datasets import build_split

    cfg = load_config(config_path)
    if not Path(cfg.data.root).exists():
        return None
    _, _, split, _ = build_split(cfg, cache_only=True)
    where = {p: name for name in ("train", "val", "test") for p, _ in getattr(split, name)}
    return cfg, where


def show_copy_check(image: np.ndarray) -> None:
    from idefect.data.duplicates import DuplicateCacheMissing, match_against

    st.subheader("Is this a copy of a NEU image?")
    try:
        idx = dataset_index()
    except DuplicateCacheMissing:
        st.info("The near-duplicate index has not been built yet. Run "
                "`python scripts/dataset_report.py` once, then try again.")
        return
    except RuntimeError as err:  # for example the dataset no longer matches its manifest
        st.error(str(err))
        return
    if idx is None:
        st.info("NEU dataset not found locally (data/raw/NEU-CLS), so the copy check is skipped.")
        return
    cfg, where = idx
    with st.spinner("Searching the dataset: whole copies at any size and crops at their "
                    "original scale, under all flips, rotations and shifts..."):
        top = match_against(image, list(where), cfg.data.duplicates, top_n=1)
    if not top:
        st.info("The image has too little texture to compare.")
        return
    m = top[0]
    name = Path(m.b).name
    split = where.get(m.b, "?")
    if m.kind == "same":
        detail = f"{m.evidence} evidence, {m.transform}"
        if m.evidence == "pixel-identical":
            detail += f", {m.overlap_px:,} identical pixels"
        st.error(f"This image is a copy of **{name}** (or of part of it) from the **{split}** "
                 f"split ({detail}). If it is in train or val, the model has effectively seen "
                 "it, so this is not an independent test.")
    elif m.kind == "review":
        st.warning(f"Possible copy of **{name}** ({split} split): the evidence is borderline.")
    elif m.kind == "unverified":
        st.info(f"No copy could be verified. The closest NEU image is {name} ({split} split), but "
                "the upload has too little fine texture to confirm or rule out a copy.")
    else:
        st.success(f"No copy found. Closest NEU image is {name}, judged distinct.")


def gradcam_overlay(predictor: Predictor, image: np.ndarray) -> np.ndarray:
    """RGB overlay of the Grad-CAM heat map (predicted class) on the image the CNN sees."""
    import torch
    from matplotlib import colormaps

    from idefect.evaluation.gradcam import GradCAM

    proc = predictor.preprocess(image)
    cams, _ = GradCAM(predictor.model)(torch.from_numpy(proc)[None, None].float())
    heat = colormaps["jet"](cams[0])[..., :3]
    base = np.repeat(np.clip(proc, 0, 1)[..., None], 3, axis=2)
    return np.clip(0.55 * base + 0.45 * heat, 0, 1)


def probability_chart(probabilities: dict) -> alt.Chart:
    """Horizontal bars in the fixed class order, one band per class."""
    df = pd.DataFrame({"class": list(probabilities), "probability": list(probabilities.values())})
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
        y=alt.Y("class:N", sort=list(probabilities), title=None),
        tooltip=["class", alt.Tooltip("probability:Q", format=".4f")],
    )


def main() -> None:
    st.title("Automated Industrial Surface Defect Analysis")
    st.caption(
        "Classical computer vision (HOG / LBP / GLCM features with classical classifiers) "
        "versus CNNs on the NEU Surface Defect Database."
    )

    artifacts = find_artifacts()
    if not artifacts:
        st.warning(
            "No trained models found in `experiments/results/`.\n\n"
            "Train at least one model first, for example:\n"
            "```\n"
            "python scripts/train_classical.py\n"
            "python scripts/train_cnn.py\n"
            "```"
        )
        return

    with st.sidebar:
        st.header("Model")
        labels = {a.name: a for a in artifacts}
        choice = st.selectbox("Trained artifact", list(labels.keys()))
        predictor = load_predictor(str(labels[choice]))
        st.write(f"**Kind:** {predictor.kind}")
        st.write(f"**Input size:** {predictor.image_size}x{predictor.image_size} px")
        metrics = load_test_metrics(labels[choice])
        if metrics:
            st.subheader("Held-out test scores")
            st.write(f"Accuracy: **{metrics['accuracy']:.4f}**")
            st.write(f"Macro F1: **{metrics['macro_f1']:.4f}**")
            if metrics.get("roc_auc_ovr_macro") is not None:
                st.write(f"ROC-AUC (OvR): **{metrics['roc_auc_ovr_macro']:.4f}**")
        compare_all = st.checkbox("Compare all trained models", value=False)
        show_cam = st.checkbox("Show where a CNN looked (Grad-CAM)", value=False,
                               help="Heat map of the image regions that raised the predicted class "
                                    "score. CNN models (.pt) only.")
        check_copy = st.checkbox(
            "Check whether the upload is a copy of a NEU image", value=False,
            help="Searches the local NEU dataset for whole-image copies (any size, flip, "
                 "rotation or brightness change) and for crops at their original scale. "
                 "Needs data/raw/NEU-CLS and the index built by scripts/dataset_report.py; "
                 "takes a few seconds.")
        st.write("**Classes:** " + ", ".join(predictor.class_names))

    uploaded = st.file_uploader(
        "Upload a surface image", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
    )
    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    original = np.array(Image.open(uploaded).convert("RGB"))
    pil = Image.open(uploaded).convert("L")
    image = np.array(pil)
    result = predictor.predict(image)
    warnings = check_input(original, predictor.reference, result.confidence, predictor.kind,
                           n_neighbors=predictor.n_neighbors)

    col1, col2, col3 = st.columns([1, 1, 1.3])
    with col1:
        st.subheader("Input")
        size = f"{image.shape[1]}x{image.shape[0]}"
        if is_colour(original):
            st.image(original, caption=f"Uploaded ({size}, colour; the models see it in grayscale)",
                     width="stretch")
        else:
            st.image(pil, caption=f"Uploaded ({size}, grayscale)", width="stretch")
    with col2:
        st.subheader("What the model sees")
        st.image(predictor.preprocess(image), caption="After resize and preprocessing",
                 clamp=True, width="stretch")
    with col3:
        st.subheader("Prediction")
        st.metric("Predicted class", result.label)
        if result.confidence is not None:
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
        if result.probabilities:
            st.altair_chart(probability_chart(result.probabilities))

    for w in warnings:
        st.warning(w)
    st.caption("All models are closed-set: they always pick one of the six NEU classes, "
               "even for images that show none of them.")

    if show_cam:
        st.subheader("Where the model looked (Grad-CAM)")
        if predictor.model is not None and labels[choice].suffix == ".pt":
            c1, _ = st.columns([1, 1.3])
            with c1:
                st.image(gradcam_overlay(predictor, image), clamp=True, width="stretch",
                         caption=f"Red: regions that raised the score of '{result.label}'. "
                                 "Grad-CAM shows where evidence was found, not why.")
        else:
            st.info("Grad-CAM applies to CNNs. Choose a .pt model in the sidebar.")

    if check_copy:
        show_copy_check(image)

    if compare_all and len(artifacts) > 1:
        st.subheader("All models on this image")
        rows = []
        for art in artifacts:
            r = load_predictor(str(art)).predict(image)
            m = load_test_metrics(art) or {}
            rows.append({
                "model": r.model_kind,
                "prediction": r.label,
                "confidence": r.confidence,
                "test macro F1": m.get("macro_f1"),
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True,
            column_config={
                "confidence": st.column_config.NumberColumn(format="%.4f"),
                "test macro F1": st.column_config.NumberColumn(format="%.4f"),
            },
        )


if __name__ == "__main__":
    main()
