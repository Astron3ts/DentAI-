# main.py
import torch
from dataset import create_dataloaders
from models import build_mobilenetv2, build_resnet50
from train import run_training

if __name__ == '__main__':
    # Path to folder containing 'train/' and 'val/' sub-directories
    DATASET_PATH  = r"D:\ML\Project\Final_dataset"
    BATCH_SIZE    = 32
    EPOCHS        = 15
    LEARNING_RATE = 0.0001

    print("Loading datasets...")
    train_loader, val_loader, classes = create_dataloaders(
        data_dir=DATASET_PATH,
        batch_size=BATCH_SIZE
    )
    print(f"Detected {len(classes)} classes: {classes}")

    # ── Device setup (shared across both training runs) ────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Training device: {str(device).upper()}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  CUDA not available — training on CPU (will be slow)")


    # ── Train MobileNetV2 ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EXPERIMENT 1 — Transfer Learning with MobileNetV2")
    print("="*55)
    mobilenet = build_mobilenetv2(num_classes=len(classes), freeze_backbone=False)
    run_training(
        model=mobilenet,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path="mobilenetv2_best.pth",
        device=device
    )

    # ── Train ResNet50 ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  EXPERIMENT 2 — Transfer Learning with ResNet50")
    print("="*55)
    resnet = build_resnet50(num_classes=len(classes), freeze_backbone=False)
    run_training(
        model=resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path="resnet50_best.pth",
        device=device
    )

    print("\n✅ Both models trained. Weights saved:")
    print("   • mobilenetv2_best.pth")
    print("   • resnet50_best.pth")