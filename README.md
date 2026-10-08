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
