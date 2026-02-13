# app.py
import streamlit as st
from PIL import Image
import torch

from utils import (
    load_model,
    pil_to_tensor,
    predict_topk,
    capture_activations,
    activation_mean,
    activation_channel,
    gradcam_overlay,
    activation_maximization,
)

st.set_page_config(layout="wide", page_title="CNN Interpretability")
st.title("CNN Interpretability")

model = load_model()

layers = {
    "Shallow (layer1)": model.layer1,
    "Deep (layer4)": model.layer4,
}


# Sidebar: Two image slots
st.sidebar.header("Inputs - Image Slots")
up_a = st.sidebar.file_uploader("Upload Image A", type=["jpg", "png", "jpeg"], key="upload_a")
up_b = st.sidebar.file_uploader("Upload Image B", type=["jpg", "png", "jpeg"], key="upload_b")

st.sidebar.divider()
layer_name = st.sidebar.selectbox("Layer for CAM", list(layers.keys()))
layer = layers[layer_name]

st.sidebar.divider()
topk = st.sidebar.slider("Top-k predictions", 1, 10, 5)

# Load images (if present)
img_a = Image.open(up_a).convert("RGB") if up_a else None
img_b = Image.open(up_b).convert("RGB") if up_b else None

if img_a is None and img_b is None:
    st.info("Upload an image in Slot A and Slot B to begin.")
    st.stop()

tabs = st.tabs(["Attribution & Activations (A vs B)", "Activation Maximization"])

def render_g_panel(pil_img: Image.Image, title: str):
    st.markdown(f"### {title}")

    x = pil_to_tensor(pil_img)

    # Input
    st.image(pil_img, caption=f"{title}: input", use_container_width=True)

    # Predictions
    preds = predict_topk(model, x, k=topk)
    st.write("Top-k predictions (class index, probability):")
    st.json(preds)

    # CAM
    st.write(f"Grad-CAM (target: {layer_name})")
    try:
        cam_img = gradcam_overlay(model, x, pil_img, layer)
        st.image(cam_img, caption=f"{title}: Grad-CAM on {layer_name}", use_container_width=True)
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")

    st.write("Activations (both layers)")

    acts = capture_activations(model, x, layers)

    c1, c2 = st.columns(2)
    for col, lname in zip([c1, c2], list(layers.keys())):
        with col:
            act = acts[lname]
            st.write(f"**{lname}**  shape: `{tuple(act.shape)}`")

            mode = st.radio(
                f"Display mode ({title} — {lname})",
                ["Mean", "Channel"],
                key=f"{title}_{lname}_mode",
                horizontal=True,
            )
            if mode == "Channel":
                ch = st.slider(
                    f"Channel ({title} — {lname})",
                    0,
                    act.shape[1] - 1,
                    0,
                    key=f"{title}_{lname}_ch",
                )
                st.image(activation_channel(act, ch).resize((224, 224)), caption=f"{lname} channel {ch}")
            else:
                st.image(activation_mean(act).resize((224, 224)), caption=f"{lname} mean over channels")


with tabs[0]:
    st.subheader("Attribution + Activations: Side-by-side comparison (Slot A vs Slot B)")

    st.markdown(
        """
**Layer motivation:**
- **Shallow (layer1)**: often edge/texture detectors (local, low-level patterns).
- **Deep (layer4)**: often more abstract/semantic patterns (object parts, shapes).

"""
    )

    if img_a is None or img_b is None:
        st.warning(
            "Required: Upload **both** Image A and Image B."
        )

    left, right = st.columns(2)

    with left:
        if img_a is not None:
            render_g_panel(img_a, "Image A")
        else:
            st.info("Upload Image A to populate this panel.")

    with right:
        if img_b is not None:
            render_g_panel(img_b, "Image B")
        else:
            st.info("Upload Image B to populate this panel.")


with tabs[1]:
    st.subheader("Activation Maximization (your own gradient ascent)")

    st.markdown(
        """
Optimize the **input image** to maximize the activation of a chosen **layer + channel/filter**.
This tab implements gradient ascent directly and lets you produce **multiple activation maps**.
"""
    )

    dummy = torch.zeros(1, 3, 224, 224, device=next(model.parameters()).device)
    dummy_acts = capture_activations(model, dummy, {"selected": layer})["selected"]
    ch_count = int(dummy_acts.shape[1])

    ch = st.slider("Channel/filter index", 0, ch_count - 1, 0)
    steps = st.slider("Steps", 50, 400, 200, step=25)
    lr = st.slider("Learning rate", 0.01, 0.30, 0.08)
    seed = st.number_input("Random seed", 0, 10000, 0, step=1)

    l2w = st.slider("L2 weight", 0.0, 1e-3, 1e-4, format="%.6f")
    tvw = st.slider("TV weight", 0.0, 1e-3, 1e-4, format="%.6f")

    if st.button("Run gradient ascent (single channel)"):
        with st.spinner("Optimizing input image..."):
            out_img, hist = activation_maximization(
                model,
                layer,
                channel=ch,
                steps=steps,
                lr=lr,
                l2_weight=l2w,
                tv_weight=tvw,
                seed=seed,
            )
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(out_img, caption=f"{layer_name} — channel {ch}", use_container_width=True)
        with c2:
            st.write("Objective history (last 10 values):")
            st.json(hist[-10:])

    st.divider()
    st.write("### Multiple activation maps (4-channel grid)")

    preset = st.selectbox("Pick 4 channels", ["0,1,2,3", "5,10,20,30", "Custom"])
    if preset == "Custom":
        custom = st.text_input("Enter 4 comma-separated channels", "0,1,2,3")
        try:
            ch_list = [int(x.strip()) for x in custom.split(",")][:4]
        except Exception:
            ch_list = [0, 1, 2, 3]
    else:
        ch_list = [int(x.strip()) for x in preset.split(",")]

    # sanitize
    ch_list = [c for c in ch_list if 0 <= c < ch_count]
    while len(ch_list) < 4:
        ch_list.append(0)

    if st.button("Run 4-channel grid"):
        imgs = []
        with st.spinner("Generating 4 activation-maximized images..."):
            for i, cidx in enumerate(ch_list):
                im, _ = activation_maximization(
                    model,
                    layer,
                    channel=int(cidx),
                    steps=steps,
                    lr=lr,
                    l2_weight=l2w,
                    tv_weight=tvw,
                    seed=seed + i,
                )
                imgs.append((cidx, im))

        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        for col, (cidx, im) in zip([r1c1, r1c2, r2c1, r2c2], imgs):
            with col:
                st.image(im, caption=f"{layer_name} — channel {cidx}", use_container_width=True)
