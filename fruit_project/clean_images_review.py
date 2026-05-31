import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_DIR = "/workspaces/fruit/fruit_project"
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
REMOVED_DIR = os.path.join(PROJECT_DIR, "data", "removed_wrong")

CLASSES = ["apple", "banana", "orange", "grape", "strawberry"]

os.makedirs(REMOVED_DIR, exist_ok=True)


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def review_class(class_name):
    class_dir = os.path.join(RAW_DIR, class_name)
    removed_class_dir = os.path.join(REMOVED_DIR, class_name)
    os.makedirs(removed_class_dir, exist_ok=True)

    if not os.path.exists(class_dir):
        print(f"文件夹不存在：{class_dir}")
        return

    files = [f for f in os.listdir(class_dir) if is_image_file(f)]
    files.sort()

    print(f"\n开始审核类别：{class_name}")
    print(f"共有图片：{len(files)} 张")
    print("操作说明：")
    print("  回车 / y = 保留")
    print("  n = 移到 removed_wrong")
    print("  q = 退出当前类别")
    print("-" * 50)

    for filename in files:
        path = os.path.join(class_dir, filename)

        if not os.path.exists(path):
            continue

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"无法打开：{filename}，自动移除。原因：{e}")
            shutil.move(path, os.path.join(removed_class_dir, filename))
            continue

        plt.imshow(img)
        plt.title(f"{class_name} | {filename}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.1)

        ans = input(f"是否保留这张 {class_name} 图片？[Enter/y保留, n移除, q退出]：").strip().lower()

        plt.close()

        if ans == "n":
            dst = os.path.join(removed_class_dir, filename)

            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                dst = os.path.join(removed_class_dir, base + "_dup" + ext)

            shutil.move(path, dst)
            print(f"已移除：{filename}")

        elif ans == "q":
            print("退出当前类别审核。")
            break

        else:
            print(f"保留：{filename}")


def main():
    print("请选择要审核的类别：")
    for i, c in enumerate(CLASSES, start=1):
        print(f"{i}. {c}")

    print("0. 全部类别")

    choice = input("请输入编号：").strip()

    if choice == "0":
        selected = CLASSES
    else:
        idx = int(choice) - 1
        selected = [CLASSES[idx]]

    for class_name in selected:
        review_class(class_name)

    print("\n审核完成。")
    print(f"被移除图片位置：{REMOVED_DIR}")


if __name__ == "__main__":
    main()