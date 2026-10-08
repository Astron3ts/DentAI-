# 🦷 Dent-AI: Dental Pathology Classification using Transfer Learning

> **Automated Multi-Class Oral Condition Screening from Intraoral Photographs**  

---

## 📌 Project Overview
**Dent-AI** is an end-to-end computer vision and explainable AI (XAI) pipeline designed to classify six common oral pathology conditions from intraoral RGB photographs:
* **Calculus**
* **Caries**
* **Gingivitis**
* **Mouth Ulcer**
* **Tooth Discoloration**
* **Hypodontia**

The system compares **MobileNetV2** (lightweight edge/mobile candidate) against **ResNet50** (deep residual candidate) under identical training and evaluation settings.

---

## 🚀 Key Highlights
* **Comparative Transfer Learning:** Evaluates MobileNetV2 vs. ResNet50 on Macro F1, balanced accuracy, latency, and model size.
* **Explainable AI (Grad-CAM):** Visualizes region-of-interest attention heatmaps over suspected oral conditions.
* **Safety & Confidence Gate:** Automatically suppresses outputs when prediction confidence is below a safety threshold.
* **0–100 Screening Index:** Converts accepted classifications into a calibrated non-clinical indicator.
* **Conservative Clinical Guidance:** Provides ethical, non-prescriptive daily hygiene roadmaps and professional referral flags.
* **Interactive Dashboard:** Full Streamlit diagnostic interface with single-model and comparative inspection modes.

---

## ⚡ Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128

# 2. Launch the web application
python -m streamlit run app.py
```

<details>
<summary>🛠️ <b>Model Training & Evaluation (For Developers)</b></summary>

```bash
# Run dataset audit
python data_audit.py

# Train both backbones
python main.py

# Run comparative benchmark
python evaluate.py
```
</details>

---

## ⚠️ Ethical & Clinical Disclaimer
*Dent-AI is intended strictly for preliminary educational screening. It does NOT provide medical diagnoses, treatment recommendations, or prescriptions. All users must consult a licensed dental professional for clinical examination and definitive diagnosis.*
