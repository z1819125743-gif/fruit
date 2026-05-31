import os
import shutil
import random
from PIL import Image
import imagehash
from tqdm import tqdm


PROJECT_DIR = "/workspaces/fruit/fruit_project"

RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
SPLIT_DIR = os.path.join(PROJECT_DIR, "data", "split")

CLASSES = ["apple", "banana", "orange", "grape", "strawberry"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RESET_SPLIT_DIR = True
HASH_DIFF_THRESHOLD = 3
RANDOM_SEED = 42


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in [
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    ]


def is_valid_image(path, min_size=160):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            w, h = img.size

        return w >= min_size and h >= min_size

    except Exception:
        return False


def clean_invalid_images(class_dir):
    removed = 0

    for file in os.listdir(class_dir):
        path = os.path.join(class_dir, file)

        if not os.path.isfile(path):
            continue

        if not is_image_file(file):
            os.remove(path)
            removed += 1
            continue

        if not is_valid_image(path):
            os.remove(path)
            removed += 1

    return removed


def remove_duplicate_images(class_dir):
    hashes = []
    removed = 0

    files = sorted([
        f for f in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, f)) and is_image_file(f)
    ])

    for file in tqdm(files, desc=f"去重 {os.path.basename(class_dir)}"):
        path = os.path.join(class_dir, file)

        try:
            with Image.open(path) as img:
                img_hash = imagehash.phash(img)

            duplicate = False

            for old_hash in hashes:
                if abs(img_hash - old_hash) <= HASH_DIFF_THRESHOLD:
                    duplicate = True
                    break

            if duplicate:
                os.remove(path)
                removed += 1
            else:
                hashes.append(img_hash)

        except Exception:
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass

    return removed


def rename_images(class_dir, class_name):
    files = sorted([
        f for f in os.listdir(class_dir)
        if os.path.isfile(os.path.join(class_dir, f)) and is_image_file(f)
    ])

    temp_dir = os.path.join(class_dir, "_temp_rename")
    os.makedirs(temp_dir, exist_ok=True)

    for idx, file in enumerate(files, start=1):
        old_path = os.path.join(class_dir, file)
        new_name = f"{class_name}_{idx:04d}.jpg"
        new_path = os.path.join(temp_dir, new_name)

        try:
            with Image.open(old_path) as img:
                img = img.convert("RGB")
                img.save(new_path, "JPEG", quality=95)
        except Exception:
            pass

    for file in files:
        old_path = os.path.join(class_dir, file)
        if os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass

    for file in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, file), os.path.join(class_dir, file))

    os.rmdir(temp_dir)


def create_split_dirs():
    if RESET_SPLIT_DIR and os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            os.makedirs(os.path.join(SPLIT_DIR, split, class_name), exist_ok=True)


def split_dataset():
    random.seed(RANDOM_SEED)
    summary = {}

    for class_name in CLASSES:
        class_dir = os.path.join(RAW_DIR, class_name)

        files = sorted([
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and is_image_file(f)
        ])

        random.shuffle(files)

        total = len(files)
        train_num = int(total * TRAIN_RATIO)
        val_num = int(total * VAL_RATIO)

        train_files = files[:train_num]
        val_files = files[train_num:train_num + val_num]
        test_files = files[train_num + val_num:]

        for split, split_files in {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }.items():
            dst_dir = os.path.join(SPLIT_DIR, split, class_name)

            for file in split_files:
                src_path = os.path.join(class_dir, file)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)

        summary[class_name] = {
            "total": total,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files)
        }

    return summary


def print_summary(summary):
    print("\n" + "=" * 60)
    print("数据集划分结果")
    print("=" * 60)
    print(f"{'类别':<15}{'总数':<10}{'训练集':<10}{'验证集':<10}{'测试集':<10}")
    print("-" * 60)

    for class_name, item in summary.items():
        print(
            f"{class_name:<15}"
            f"{item['total']:<10}"
            f"{item['train']:<10}"
            f"{item['val']:<10}"
            f"{item['test']:<10}"
        )


def main():
    print("开始清洗、去重并划分数据集...")

    for class_name in CLASSES:
        class_dir = os.path.join(RAW_DIR, class_name)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"找不到类别文件夹：{class_dir}")

        print("\n" + "=" * 60)
        print(f"处理类别：{class_name}")
        print("=" * 60)

        before_count = len([
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and is_image_file(f)
        ])

        print(f"初始图片数量：{before_count}")

        invalid_removed = clean_invalid_images(class_dir)
        print(f"删除无效图片数量：{invalid_removed}")

        duplicate_removed = remove_duplicate_images(class_dir)
        print(f"删除重复/近似重复图片数量：{duplicate_removed}")

        rename_images(class_dir, class_name)

        after_count = len([
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f)) and is_image_file(f)
        ])

        print(f"清洗后图片数量：{after_count}")

        if after_count < 100:
            print(f"警告：{class_name} 清洗后不足100张，建议补充图片。")

    create_split_dirs()
    summary = split_dataset()
    print_summary(summary)

    print("\n处理完成！")
    print(f"划分后的数据集位于：{SPLIT_DIR}")


if __name__ == "__main__":
    main()