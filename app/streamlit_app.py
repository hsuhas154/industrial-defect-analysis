#!/usr/bin/env python3
"""Streamlit inference UI for the Industrial Surface Defect Analysis project.

Upload a steel-surface image, pick a trained model (custom CNN or a classical
HOG + Decision Tree / SVM baseline), and view the predicted defect class with
per-class confidences. Optionally run every trained model on the same image
to compare them side by side.

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

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


def main() -> None:
    st.title("Automated Industrial Surface Defect Analysis")
    st.caption(
        "Classical CV (HOG + Decision Tree / SVM) versus a custom CNN "
        "on the NEU Surface Defect Database."
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
        st.write("**Classes:** " + ", ".join(predictor.class_names))

    uploaded = st.file_uploader(
        "Upload a surface image", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
    )
    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    pil = Image.open(uploaded).convert("L")
    image = np.array(pil)
    result = predictor.predict(image)

    col1, col2, col3 = st.columns([1, 1, 1.3])
    with col1:
        st.subheader("Input")
        st.image(pil, caption=f"Uploaded ({image.shape[1]}x{image.shape[0]}, grayscale)",
                 width="stretch")
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
            probs = pd.DataFrame(
                {"class": list(result.probabilities), "probability": list(result.probabilities.values())}
            ).sort_values("probability", ascending=False)
            st.bar_chart(probs, x="class", y="probability", horizontal=True)

    if compare_all and len(artifacts) > 1:
        st.subheader("All models on this image")
        rows = []
        for art in artifacts:
            r = load_predictor(str(art)).predict(image)
            m = load_test_metrics(art) or {}
            rows.append({
                "model": r.model_kind,
                "prediction": r.label,
                "confidence": None if r.confidence is None else round(r.confidence, 4),
                "test macro F1": m.get("macro_f1"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)


if __name__ == "__main__":
    main()
