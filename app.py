"""
CS464 – CAE Anomaly Detection Web App
Streamlit demo: upload any fashion image → get anomaly score + heatmap
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CAE Anomaly Detector",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Fashion Anomaly Detector")
st.markdown(
    "**CS464 Deep Learning – Term Project**  \n"
    "Upload a fashion image to check if it looks like a **T-shirt (normal)** "
    "or something else **(anomaly)**."
)
st.divider()

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "improved_cae.keras")
    return keras.models.load_model(model_path)

with st.spinner("Loading model..."):
    model = load_model()

# Threshold computed from training (95th percentile of normal test errors)
THRESHOLD = 0.000610

# ── Sidebar info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown(
        "1. The **Improved CAE** was trained **only on T-shirt images**.\n"
        "2. It learns to reconstruct T-shirts with very low error.\n"
        "3. Non-T-shirt items produce **high reconstruction error**.\n"
        "4. Anything above the threshold → **Anomaly**."
    )
    st.metric("Anomaly Threshold (MSE)", f"{THRESHOLD:.6f}")
    st.markdown("---")
    st.header("📊 Model Info")
    st.markdown(
        "- **Architecture**: Improved CAE\n"
        "- **BatchNorm** + **LeakyReLU** + **Skip connections**\n"
        "- **Val Loss**: 0.000291\n"
        "- **ROC-AUC**: 0.8137\n"
        "- **Normal class**: T-shirt/top\n"
        "- **Input size**: 28 × 28 greyscale"
    )
    st.markdown("---")
    st.caption("CS464 Deep Learning · Spring 2026")

# ── Fashion-MNIST class names for reference ───────────────────────────────────
CLASS_NAMES = [
    "T-shirt/top (Normal)", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ── Helper: preprocess uploaded image ────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img_grey = img.convert("L")                     # to greyscale
    img_resized = img_grey.resize((28, 28), Image.LANCZOS)
    arr = np.array(img_resized, dtype="float32") / 255.0
    return arr.reshape(1, 28, 28, 1)

# ── Helper: error heatmap as PIL image ───────────────────────────────────────
def make_heatmap(original: np.ndarray, reconstructed: np.ndarray) -> Image.Image:
    error_map = np.abs(original.squeeze() - reconstructed.squeeze())
    normed = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    rgba = cm.hot(normed)                           # hot colourmap → RGBA
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((224, 224), Image.NEAREST)

# ── Helper: upscale 28×28 for display ────────────────────────────────────────
def upscale(arr: np.ndarray, size=224) -> Image.Image:
    img_u8 = (arr.squeeze() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_u8, mode="L").resize((size, size), Image.NEAREST)

# ── Main: try sample images from Fashion-MNIST ───────────────────────────────
st.subheader("Try it with Fashion-MNIST samples")

@st.cache_resource
def load_samples():
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    return x_test, y_test

x_test, y_test = load_samples()

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_class = st.selectbox(
        "Pick a class to test:",
        options=list(range(10)),
        format_func=lambda i: CLASS_NAMES[i]
    )
with col_sel2:
    sample_idx_in_class = st.number_input("Sample index", min_value=0, max_value=49, value=0)

if st.button("Run on this sample", type="primary"):
    class_idxs = np.where(y_test == selected_class)[0]
    chosen_idx = class_idxs[sample_idx_in_class]
    raw_img = x_test[chosen_idx].astype("float32") / 255.0
    inp = raw_img.reshape(1, 28, 28, 1)

    recon = model.predict(inp, verbose=0)
    mse = float(np.mean(np.square(inp - recon)))
    is_anomaly = mse > THRESHOLD

    st.divider()
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.image(upscale(raw_img), caption="Original", use_container_width=True)
    with r2:
        st.image(upscale(recon[0]), caption="Reconstructed", use_container_width=True)
    with r3:
        st.image(make_heatmap(inp, recon), caption="Error Heatmap", use_container_width=True)
    with r4:
        st.metric("Reconstruction Error (MSE)", f"{mse:.6f}")
        st.metric("Threshold", f"{THRESHOLD:.6f}")
        if is_anomaly:
            st.error(f"🚨 **ANOMALY** — {CLASS_NAMES[selected_class]}")
        else:
            st.success(f"✅ **NORMAL** — {CLASS_NAMES[selected_class]}")

st.divider()

# ── Main: upload your own image ───────────────────────────────────────────────
st.subheader("Or upload your own image")
uploaded = st.file_uploader(
    "Upload any greyscale or colour image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    pil_img = Image.open(uploaded)
    inp = preprocess(pil_img)

    recon = model.predict(inp, verbose=0)
    mse = float(np.mean(np.square(inp - recon)))
    is_anomaly = mse > THRESHOLD

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(pil_img.resize((224, 224)), caption="Your Image (resized)", use_container_width=True)
    with c2:
        st.image(upscale(recon[0]), caption="Reconstructed (28×28)", use_container_width=True)
    with c3:
        st.image(make_heatmap(inp, recon), caption="Error Heatmap", use_container_width=True)
    with c4:
        st.metric("Reconstruction Error (MSE)", f"{mse:.6f}")
        st.metric("Threshold", f"{THRESHOLD:.6f}")
        if is_anomaly:
            st.error("🚨 **ANOMALY DETECTED**")
        else:
            st.success("✅ **NORMAL (T-shirt-like)**")
