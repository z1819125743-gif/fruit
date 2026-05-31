import os
import copy
import time
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 配置区域
# =========================

PROJECT_DIR = "/workspaces/fruit/fruit_project"

DATA_DIR = os.path.join(PROJECT_DIR, "data", "split")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = "best_resnet18_weight_decay_1e4.pth"
CLASS_TO_IDX_PATH = os.path.join(OUTPUT_DIR, "class_to_idx.json")
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")

NUM_CLASSES = 5
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# 如果电脑没有 GPU，会自动使用 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 固定随机种子
# =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# 数据加载
# =========================

def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    with open(CLASS_TO_IDX_PATH, "w", encoding="utf-8") as f:
        json.dump(train_dataset.class_to_idx, f, ensure_ascii=False, indent=4)

    print("类别映射：")
    print(train_dataset.class_to_idx)

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    dataset_sizes = {
        "train": len(train_dataset),
        "val": len(val_dataset),
        "test": len(test_dataset)
    }

    class_names = train_dataset.classes

    return dataloaders, dataset_sizes, class_names


# =========================
# 构建模型
# =========================
def build_model():
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("已加载 ImageNet 预训练权重。")
    except Exception:
        model = models.resnet18(pretrained=True)
        print("已加载 ImageNet 预训练权重。")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    model = model.to(DEVICE)
    return model


# =========================
# 训练与验证
# =========================

def train_model(model, dataloaders, dataset_sizes):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print("\n" + "-" * 60)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 60)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                MODEL_SAVE_PATH = "best_resnet18_label_smoothing_01.pth"
                print(f"保存当前最佳模型，val_acc = {best_val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\n训练完成，用时：{total_time // 60:.0f}分 {total_time % 60:.0f}秒")
    print(f"最佳验证集准确率：{best_val_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model, history


# =========================
# 测试模型
# =========================

def evaluate_model(model, dataloaders, class_names):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )

    cm = confusion_matrix(all_labels, all_preds)

    print("\n测试集分类报告：")
    print(report)

    print("混淆矩阵：")
    print(cm)

    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n测试集准确率：{test_acc:.4f}")


# =========================
# 绘制训练曲线
# =========================

def plot_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=300)
    print(f"训练曲线已保存到：{CURVE_PATH}")


# =========================
# 主程序
# =========================

def main():
    print("当前设备：", DEVICE)

    set_seed(RANDOM_SEED)

    dataloaders, dataset_sizes, class_names = get_dataloaders()

    print("数据集大小：")
    print(dataset_sizes)

    model = build_model()

    model, history = train_model(
        model,
        dataloaders,
        dataset_sizes
    )

    evaluate_model(
        model,
        dataloaders,
        class_names
    )

    plot_curves(history)

    print("\n全部完成！")
    print(f"最佳模型保存路径：{MODEL_SAVE_PATH}")
    print(f"类别映射保存路径：{CLASS_TO_IDX_PATH}")


if __name__ == "__main__":
    main()
