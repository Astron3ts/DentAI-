import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)

from dataset import create_dataloaders
from models import build_mobilenetv2, build_resnet50

def bootstrap_macro_f1(y_true, y_pred, n_bootstraps: int = 1000, alpha: float = 0.05):
    """Calculates 95% bootstrap confidence intervals for Macro F1."""
    rng = np.random.default_rng(42)
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        bootstrapped_scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    low = np.percentile(bootstrapped_scores, (alpha / 2.0) * 100)
    high = np.percentile(bootstrapped_scores, (1.0 - alpha / 2.0) * 100)
    return low, high

def benchmark_latency(model, sample_input, device, runs: int = 100):
    """Measures median inference latency per sample in milliseconds."""
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(sample_input)
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0) # ms

    return float(np.median(timings))

def evaluate_backbone(model_name: str, model_builder, weight_path: str, val_loader, classes, device):
    """Evaluates a single model backbone across all project metrics."""
    if not os.path.exists(weight_path):
        print(f"Skipping {model_name}: weights not found at {weight_path}")
        return None

    model = model_builder(num_classes=len(classes), freeze_backbone=False)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    ci_low, ci_high = bootstrap_macro_f1(all_labels, all_preds)

    # Benchmark latency on 1 sample
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    latency_ms = benchmark_latency(model, dummy_input, device)

    # Count parameters & file size
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    filesize_mb = os.path.getsize(weight_path) / (1024 * 1024)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Normalized Confusion Matrix — {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_filename = f"confusion_matrix_{model_name.lower()}.png"
    plt.savefig(cm_filename, dpi=300)
    plt.close()
    print(f"Saved {model_name} confusion matrix to {cm_filename}")

    # Detailed report
    print(f"\n--- {model_name} Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes, digits=4, zero_division=0))

    return {
        "Architecture": model_name,
        "Macro F1": f"{macro_f1:.4f}",
        "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "Balanced Acc": f"{bal_acc:.4f}",
        "Latency (ms)": f"{latency_ms:.2f} ms",
        "Parameters (M)": f"{params_m:.2f} M",
        "Model Size": f"{filesize_mb:.1f} MB",
    }

def run_comparative_evaluation(data_dir: str):
    """Runs locked evaluation comparing MobileNetV2 and ResNet50."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {str(device).upper()}")

    _, val_loader, classes = create_dataloaders(data_dir=data_dir, batch_size=32)

    candidates = [
        ("MobileNetV2", build_mobilenetv2, "mobilenetv2_best.pth"),
        ("ResNet50", build_resnet50, "resnet50_best.pth"),
    ]

    records = []
    for name, builder, weight in candidates:
        res = evaluate_backbone(name, builder, weight, val_loader, classes, device)
        if res:
            records.append(res)

    if records:
        comparison_df = pd.DataFrame(records)
        print("\n" + "=" * 75)
        print("                 MODEL COMPARISON TABLE (RQ1)")
        print("=" * 75)
        print(comparison_df.to_string(index=False))
        print("=" * 75)
        comparison_df.to_csv("model_comparison_results.csv", index=False)
        print("Saved comparison table to: model_comparison_results.csv\n")

if __name__ == "__main__":
    DATASET_PATH = r"D:\ML\Project\Final_dataset"
    run_comparative_evaluation(data_dir=DATASET_PATH)