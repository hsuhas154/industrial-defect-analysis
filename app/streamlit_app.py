#!/usr/bin/env python3
"""Streamlit inference UI for the Industrial Surface Defect Analysis prototype.

Upload a steel-surface image, pick a trained model (custom CNN or a classical
HOG + Decision Tree / SVM baseline), and view the predicted defect class with
per-class confidences.

Run with:
    streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

# Make src/ importable when launched via `streamlit run` from the repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from idefect.inference.predict import Predictor  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "experiments" / "results"

st.set_page_config(page_title="Industrial Surface Defect Analysis", layout="wide")


@st.cache_resource(show_spinner=False)
def load_predictor(artifact_path: str) -> Predictor:
    return Predictor(artifact_path)


def find_artifacts():
    if not RESULTS_DIR.exists():
        return []
    arts = sorted(RESULTS_DIR.glob("*.joblib")) + sorted(RESULTS_DIR.glob("*.pt"))
    return arts


def main() -> None:
    st.title("Automated Industrial Surface Defect Analysis")
    st.caption(
        "Phase 1 prototype — classical CV (HOG + Decision Tree / SVM) vs a custom CNN "
        "on the NEU Surface Defect Database."
    )

    artifacts = find_artifacts()
    if not artifacts:
        st.warning(
            "No trained models found in `experiments/results/`.\n\n"
            "Train at least one model first, for example:\n"
            "```\n"
            "python scripts/generate_dummy_data.py --out data/raw/NEU-CLS --per-class 40\n"
            "python scripts/train_classical.py\n"
            "python scripts/train_cnn.py\n"
            "```"
        )
        return

    with st.sidebar:
        st.header("Model")
        labels = {a.name: str(a) for a in artifacts}
        choice = st.selectbox("Trained artifact", list(labels.keys()))
        predictor = load_predictor(labels[choice])
        st.write(f"**Kind:** {predictor.kind}")
        st.write(f"**Classes:** {', '.join(predictor.class_names)}")
        st.write(f"**Input size:** {predictor.image_size}×{predictor.image_size}")

    uploaded = st.file_uploader(
        "Upload a surface image", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"]
    )
    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    pil = Image.open(uploaded).convert("L")
    image = np.array(pil)

    result = predictor.predict(image)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input")
        st.image(pil, caption="Uploaded image (grayscale)", use_container_width=True)
    with col2:
        st.subheader("Prediction")
        st.metric("Predicted class", result.label)
        if result.confidence is not None:
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
        if result.probabilities:
            st.subheader("Per-class confidence")
            probs = dict(
                sorted(result.probabilities.items(), key=lambda kv: -kv[1])
            )
            st.bar_chart(probs)

    st.caption(
        "Note: predictions from a model trained on synthetic smoke-test data are "
        "not meaningful. Train on the real NEU dataset for real results."
    )


if __name__ == "__main__":
    main()
