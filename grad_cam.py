import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import build_mobilenetv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def run_gradcam(image_path: str, model_path: str, classes: list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_mobilenetv2(num_classes=len(classes), freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    raw_image = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb_f32   = np.float32(raw_image) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs     = torch.nn.functional.softmax(model(tensor)[0], dim=0)
        pred_idx  = probs.argmax().item()
        pred_conf = probs[pred_idx].item() * 100

    cam           = GradCAM(model=model, target_layers=[model.features[-1]])
    grayscale_cam = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred_idx)])[0]
    cam_image     = show_cam_on_image(rgb_f32, grayscale_cam, use_rgb=True)

    true_label = os.path.basename(os.path.dirname(image_path))
    print(f"Image:     {image_path}")
    print(f"True:      {true_label}")
    print(f"Predicted: {classes[pred_idx]}  ({pred_conf:.2f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(raw_image);   ax1.set_title(f"Original\nTrue: {true_label}");          ax1.axis("off")
    ax2.imshow(cam_image);   ax2.set_title(f"Grad-CAM\n{classes[pred_idx]} ({pred_conf:.1f}%)"); ax2.axis("off")
    plt.tight_layout()
    plt.savefig("gradcam_output.png", dpi=300)
    print("Saved → gradcam_output.png")
    plt.show()

if __name__ == "__main__":
    MODEL_CHECKPOINT = "mobilenetv2_best.pth"
    VAL_DIR = r"D:\ML\Project\Final_dataset\val"
    CLASSES = ["Calculus", "Caries", "Gingivitis", "Mouth Ulcer", "Tooth Discoloration", "hypodontia"]

    all_images = [
        os.path.join(root, f)
        for root, _, files in os.walk(VAL_DIR)
        for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not all_images:
        print(f"No images found in {VAL_DIR}")
    else:
        run_gradcam(random.choice(all_images), MODEL_CHECKPOINT, CLASSES)