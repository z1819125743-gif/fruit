import os
import json
import shutil

from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


PROJECT_DIR = "/workspaces/fruit/fruit_project"

TEST_DIR = os.path.join(PROJECT_DIR, "data", "external_test")
MODEL_PATH = os.path.join(PROJECT_DIR, "outputs", "best_resnet18_fruit.pth")
CLASS_TO_IDX_PATH = os.path.join(PROJECT_DIR, "outputs", "class_to_idx.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "wrong_predictions")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def load_class_mapping():
    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

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


def predict_image(model, image_path, transform, idx_to_class):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    return idx_to_class[pred_idx], confidence


def safe_copy(src_path, dst_dir, filename):
    os.makedirs(dst_dir, exist_ok=True)

    base, ext = os.path.splitext(filename)
    dst_path = os.path.join(dst_dir, filename)

    count = 1
    while os.path.exists(dst_path):
        dst_path = os.path.join(dst_dir, f"{base}_{count}{ext}")
        count += 1

    shutil.copy2(src_path, dst_path)


def main():
    if not os.path.exists(TEST_DIR):
        print(f"错误：找不到外部测试集目录：{TEST_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_to_idx, idx_to_class = load_class_mapping()
    model = build_model(len(class_to_idx))
    transform = get_transform()

    total = 0
    wrong = 0

    print(f"开始收集错误预测图片：{TEST_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")
    print("=" * 60)

    for true_class in sorted(os.listdir(TEST_DIR)):
        class_dir = os.path.join(TEST_DIR, true_class)

        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if not is_image_file(filename):
                continue

            total += 1
            src_path = os.path.join(class_dir, filename)

            try:
                pred_class, confidence = predict_image(
                    model,
                    src_path,
                    transform,
                    idx_to_class
                )
            except Exception as e:
                print(f"跳过无法读取图片：{src_path}，原因：{e}")
                continue

            if pred_class != true_class:
                wrong += 1

                save_name = (
                    f"true-{true_class}_pred-{pred_class}_"
                    f"conf-{confidence:.2f}_{filename}"
                )

                dst_dir = os.path.join(OUTPUT_DIR, f"true-{true_class}", f"pred-{pred_class}")
                safe_copy(src_path, dst_dir, save_name)

                print(
                    f"错误：真实={true_class}, 预测={pred_class}, "
                    f"置信度={confidence:.2%}, 文件={filename}"
                )

    print("\n" + "=" * 60)
    print("收集完成")
    print(f"总图片数：{total}")
    print(f"错误数：{wrong}")

    if total > 0:
        print(f"错误率：{wrong / total:.2%}")

    print(f"错误图片目录：{OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()