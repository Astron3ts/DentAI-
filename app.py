import io
import os
import time
import torch
import numpy as np
import streamlit as st
from PIL import Image, ImageStat
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import build_mobilenetv2, build_resnet50
from recommend import generate_guidance, screening_index, RISK_ICONS

st.set_page_config(
    page_title="Dent-AI Dashboard",
    page_icon="🦷",
    layout="wide"
)

CLASSES = [
    'Calculus', 'Caries', 'Gingivitis',
    'Mouth Ulcer', 'Tooth Discoloration', 'hypodontia'
]

MODEL_CONFIGS = {
    "MobileNetV2": {
        "builder": build_mobilenetv2,
        "weight": "mobilenetv2_best.pth",
        "cam_layer": lambda m: [m.features[-1]],
    },
    "ResNet50": {
        "builder": build_resnet50,
        "weight": "resnet50_best.pth",
        "cam_layer": lambda m: [m.layer4[-1]],
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def check_image_quality(image: Image.Image):
    grayscale = image.convert('L')
    stat = ImageStat.Stat(grayscale)
    mean_brightness = stat.mean[0]
    std_contrast = stat.stddev[0]

    if mean_brightness < 25.0:
        return False, "Image rejected: Photograph is too dark (underexposed). Please retake under adequate direct lighting."
    if mean_brightness > 240.0:
        return False, "Image rejected: Photograph is overexposed / washed out. Please reduce direct glare or flash."
    if std_contrast < 12.0:
        return False, "Image rejected: Low contrast or blank image detected. Please provide a clear, in-focus close-up photograph."
    return True, "Quality check passed."

@st.cache_resource
def load_model(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    weight = cfg["weight"]
    if not os.path.exists(weight):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cfg["builder"](num_classes=len(CLASSES), freeze_backbone=False)
    state = torch.load(weight, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device

@st.cache_data(show_spinner=False)
def run_inference(_model, _device, model_name: str, img_bytes: bytes):
    raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_224 = raw_img.resize((224, 224))
    rgb_f32 = np.asarray(img_224, dtype=np.float32) / 255.0

    tensor = PREPROCESS(img_224).unsqueeze(0).to(_device)

    # Latency timed execution
    if _device.type == "cuda":
        torch.cuda.synchronize()
    start_t = time.perf_counter()

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.nn.functional.softmax(logits[0], dim=0).cpu().numpy()

    if _device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    # Top-2 ranking
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top2_idx = int(sorted_indices[1])

    # Explainability: Grad-CAM
    target_layers = MODEL_CONFIGS[model_name]["cam_layer"](_model)
    with GradCAM(model=_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(top1_idx)])[0]
    cam_image = show_cam_on_image(rgb_f32, grayscale_cam, use_rgb=True)

    return {
        "pred_idx": top1_idx,
        "pred_class": CLASSES[top1_idx],
        "confidence": float(probs[top1_idx]) * 100.0,
        "runner_up_class": CLASSES[top2_idx],
        "runner_up_conf": float(probs[top2_idx]) * 100.0,
        "probs": probs,
        "cam_img": cam_image,
        "latency_ms": latency_ms
    }

# Session state initialization
st.session_state.setdefault("results", {})
st.session_state.setdefault("analyzed", False)
st.session_state.setdefault("current_file_id", None)

# Header
st.title("🦷 Dent-AI: AI Pathology Screening & Comparison")
st.markdown(
    "Automated Multi-Class Oral Condition Screening from Intraoral Photographs using Transfer Learning. "
    "Designed for honest uncertainty, transparent Grad-CAM explainability, and safe non-clinical guidance."
)

# Sidebar
st.sidebar.header("⚙️ System Configuration")

available_models = [name for name, cfg in MODEL_CONFIGS.items() if os.path.exists(cfg["weight"])]

if not available_models:
    st.error("❌ No trained model weights found in the project directory. Train models first using `python main.py`.")
    st.stop()

if len(available_models) > 1:
    mode = st.sidebar.radio(
        "Inference Mode",
        ["🔀 Compare Both Models", "🎯 Single Model"],
        index=0
    )
else:
    mode = "🎯 Single Model"
    st.sidebar.info(f"Only `{available_models[0]}` weight file found.")

if mode == "🎯 Single Model":
    selected_model = st.sidebar.selectbox("Select Backbone Architecture", available_models)
    active_models = [selected_model]
else:
    active_models = available_models

confidence_threshold = st.sidebar.slider("Confidence Gate Threshold (%)", 50, 95, 70, help="Predictions below this threshold are rejected to prevent unsafe diagnosis.")
loaded = {}
for name in active_models:
    with st.spinner(f"Loading {name}..."):
        m, d = load_model(name)
    if m is not None:
        loaded[name] = (m, d)

# Image Upload
st.markdown("---")
st.subheader("📸 Step 1: Upload Intraoral Photograph")
uploaded_file = st.file_uploader("Upload a close-up oral photograph (JPEG or PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["current_file_id"] != file_id:
        st.session_state["current_file_id"] = file_id
        st.session_state["analyzed"] = False
        st.session_state["results"] = {}

    raw_image = Image.open(uploaded_file).convert("RGB")
    img_col, btn_col = st.columns([2, 1])

    with img_col:
        st.image(raw_image, caption="Uploaded Intraoral Photo", width=340)

    # Step 1 Quality Gate
    is_valid, quality_msg = check_image_quality(raw_image)

    with btn_col:
        if not is_valid:
            st.error(f"⚠️ {quality_msg}")
            run_btn = False
        else:
            st.success("✅ Image quality check passed.")
            run_btn = st.button("🔍 Run AI Screening & Explainability", type="primary", use_container_width=True)

    if run_btn:
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        results = {}

        for name, (model, device) in loaded.items():
            with st.spinner(f"Running inference with {name}..."):
                res = run_inference(model, device, name, img_bytes)
                results[name] = res

        st.session_state["results"] = results
        st.session_state["analyzed"] = True
else:
    st.session_state["analyzed"] = False
    st.session_state["results"] = {}
    st.session_state["current_file_id"] = None

# Step 2: Diagnostic Results & Grad-CAM
if st.session_state["analyzed"] and st.session_state["results"]:
    results = st.session_state["results"]
    names = list(results.keys())

    st.markdown("---")
    st.subheader("📊 Step 2: Diagnostic Report & Model Comparison")

    report_cols = st.columns(len(names))
    for col, name in zip(report_cols, names):
        r = results[name]
        conf = r["confidence"]
        is_accepted = conf >= confidence_threshold
        s_index = screening_index(conf, r["pred_class"]) if is_accepted else None

        with col:
            st.markdown(f"### {name}")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Inference Latency", f"{r['latency_ms']:.1f} ms")
            with m2:
                if is_accepted:
                    st.metric("Screening Index", f"{s_index} / 100")
                else:
                    st.metric("Screening Index", "Rejected")

            if not is_accepted:
                st.error(f"⛔ **Confidence Rejection Gate Triggered:** Model confidence **{conf:.1f}%** is below the **{confidence_threshold}%** safety threshold. Prediction rejected to prevent false reassurance.")
            else:
                st.metric(label="Primary Visible Pattern", value=r["pred_class"], delta=f"{conf:.1f}% Confidence")
                st.caption(f"Top alternative: **{r['runner_up_class']}** ({r['runner_up_conf']:.1f}%)")

            st.markdown("**Grad-CAM Region of Interest:**")
            st.image(r["cam_img"], caption=f"{name} Attention: {r['pred_class']}", use_container_width=True)

            st.markdown("**Class Probability Breakdown:**")
            prob_dict = {CLASSES[i]: float(r["probs"][i]) for i in range(len(CLASSES))}
            st.bar_chart(prob_dict, height=200)

    if len(names) == 2:
        st.markdown("---")
        r0, r1 = results[names[0]], results[names[1]]
        if r0["pred_class"] == r1["pred_class"]:
            avg_conf = (r0["confidence"] + r1["confidence"]) / 2
            st.success(f"✅ **Backbone Consensus:** Both MobileNetV2 and ResNet50 concordantly identify **{r0['pred_class']}** (Mean confidence: {avg_conf:.1f}%).")
        else:
            st.warning(
                f"⚠️ **Backbone Disagreement:** "
                f"{names[0]} flagged **{r0['pred_class']}** ({r0['confidence']:.1f}%) vs "
                f"{names[1]} flagged **{r1['pred_class']}** ({r1['confidence']:.1f}%). "
                "Clinical differential review recommended."
            )

    best_name = max(results, key=lambda n: results[n]["confidence"])
    best = results[best_name]
    pred_class = best["pred_class"]
    conf = best["confidence"]

    st.markdown("---")
    st.subheader("📋 Step 3: Conservative Hygiene Roadmap & Action Plan")

    if conf >= confidence_threshold:
        st.caption(f"Guidance derived from highest-confidence accepted output: **{best_name}** ({pred_class})")
        guidance = generate_guidance(pred_class)
        risk_icon = RISK_ICONS.get(guidance['risk'], '⚪')

        g1, g2, g3 = st.columns(3)
        g1.info(f"**{risk_icon} Severity Risk Level:**\n\n{guidance['risk']}\n\n*Summary:* {guidance['summary']}")
        g2.warning(f"**🦷 Recommended Action:**\n\n{guidance['action']}\n\n**Referral Flag:** {guidance['referral_flag']}")
        g3.success(f"**✅ Daily Hygiene Routine:**\n\n{guidance['hygiene']}")

        st.caption("⚠️ **Ethical Disclaimer:** Dent-AI provides pre-screening assessment only. It does not provide prescriptions, medical diagnoses, or replacement for examination by a licensed dental professional.")
    else:
        st.info("ℹ️ Clinical recommendations are withheld because model confidence was rejected by the safety gate. Please capture a clearer, well-lit image or consult a dental clinic.")