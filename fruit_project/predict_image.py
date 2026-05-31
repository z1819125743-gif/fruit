import os
import json
import argparse
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


PROJECT_DIR = "/workspaces/fruit/fruit_project"
MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_resnet18_fruit.pth")
CLASS_TO_IDX_PATH = os.path.join(PROJECT_DIR, "outputs", "class_to_idx.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class


def build_model(num_classes):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    return model


def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片：{image_path}")

    idx_to_class = load_class_names()
    num_classes = len(idx_to_class)

    model = build_model(num_classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()
    pred_class = idx_to_class[pred_idx]

    print("=" * 50)
    print(f"图片路径：{image_path}")
    print(f"预测类别：{pred_class}")
    print(f"置信度：{confidence * 100:.2f}%")
    print("=" * 50)

    print("\n各类别概率：")
    for idx, prob in enumerate(probs[0]):
        class_name = idx_to_class[idx]
        print(f"{class_name:<15}: {prob.item() * 100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="水果图片分类预测")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="要预测的图片路径"
    )

    args = parser.parse_args()
    predict(args.image)


if __name__ == "__main__":
    main()