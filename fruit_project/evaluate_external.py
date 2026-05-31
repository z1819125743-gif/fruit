import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_DIR = "/workspaces/fruit/fruit_project"

EXTERNAL_DIR = os.path.join(PROJECT_DIR, "data", "external_test")
MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_resnet18_fruit.pth")
CLASS_TO_IDX_PATH = os.path.join(PROJECT_DIR, "outputs", "class_to_idx.json")

NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    return class_names


def build_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in [
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    ]


def main():
    class_names = load_class_names()
    model = build_model()
    transform = get_transform()

    all_labels = []
    all_preds = []
    all_confidences = []

    print("开始评估 external_test ...\n")

    for true_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(EXTERNAL_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"跳过，文件夹不存在：{class_dir}")
            continue

        files = [
            f for f in os.listdir(class_dir)
            if is_image_file(f)
        ]

        for file in files:
            path = os.path.join(class_dir, file)

            try:
                img = Image.open(path).convert("RGB")
                x = transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = model(x)
                    probs = torch.softmax(outputs, dim=1)[0]
                    pred_idx = torch.argmax(probs).item()
                    confidence = probs[pred_idx].item()

                all_labels.append(true_idx)
                all_preds.append(pred_idx)
                all_confidences.append(confidence)

                if pred_idx != true_idx:
                    print(
                        f"预测错误: {path} | "
                        f"真实={class_name}, "
                        f"预测={class_names[pred_idx]}, "
                        f"置信度={confidence:.2%}"
                    )

            except Exception as e:
                print(f"读取失败：{path}, 原因：{e}")

    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    avg_conf = np.mean(all_confidences)

    print("\n" + "=" * 60)
    print(f"external_test 准确率：{acc:.4f}")
    print(f"external_test 平均置信度：{avg_conf:.4f}")
    print("=" * 60)

    print("\n分类报告：")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    ))

    print("混淆矩阵：")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()