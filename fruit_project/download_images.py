import os
import time
from icrawler.builtin import BingImageCrawler
from PIL import Image


# =========================
# 配置区域
# =========================

# 如果你的 download_images.py 放在 fruit_project 文件夹里面，用 "."
# 如果你的 download_images.py 放在 fruit_project 外面，用 "fruit_project"
PROJECT_DIR = "."

RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
EXTERNAL_TEST_DIR = os.path.join(PROJECT_DIR, "data", "external_test")

# 主数据集：每类目标下载数量
# 为了保证人工删除后仍不少于100张，这里每类下载120张
RAW_NUM_PER_CLASS = 120

# 外部测试集：每类下载数量
# 用于测试泛化能力
EXTERNAL_NUM_PER_CLASS = 30

# 类别与搜索关键词
# 每类设计多个关键词，减少图片重复和风格单一问题
CLASSES = {
    "apple": {
        "cn": "苹果",
        "raw_keywords": [
            "real apple fruit photo on table",
            "fresh red apple fruit real photo",
            "green apple fruit real photo",
            "apple fruit supermarket real photo",
            "苹果 水果 实物照片",
            "苹果 水果 桌面 实拍",
            "红苹果 水果 实拍",
            "青苹果 水果 实拍"
        ],
        "external_keywords": [
            "apple fruit natural light photo",
            "apple fruit market real photo",
            "apple fruit kitchen photo",
            "苹果 水果 超市 实拍",
            "苹果 水果 自然光 实拍"
        ]
    },

    "banana": {
        "cn": "香蕉",
        "raw_keywords": [
            "real banana fruit photo on table",
            "fresh banana fruit real photo",
            "yellow banana fruit real photo",
            "banana fruit supermarket real photo",
            "香蕉 水果 实物照片",
            "香蕉 水果 桌面 实拍",
            "香蕉 水果 超市 实拍",
            "黄色香蕉 实拍"
        ],
        "external_keywords": [
            "banana fruit natural light photo",
            "banana fruit market real photo",
            "banana fruit kitchen photo",
            "香蕉 水果 自然光 实拍",
            "香蕉 水果 市场 实拍"
        ]
    },

    "orange": {
        "cn": "橙子",
        "raw_keywords": [
            "real orange fruit photo on table",
            "fresh orange fruit real photo",
            "orange fruit supermarket real photo",
            "orange fruit natural light photo",
            "橙子 水果 实物照片",
            "橙子 水果 桌面 实拍",
            "橙子 水果 超市 实拍",
            "新鲜橙子 实拍"
        ],
        "external_keywords": [
            "orange fruit market real photo",
            "orange fruit kitchen photo",
            "fresh oranges natural light photo",
            "橙子 水果 自然光 实拍",
            "橙子 水果 市场 实拍"
        ]
    },

    "grape": {
        "cn": "葡萄",
        "raw_keywords": [
            "real grape fruit photo on table",
            "fresh grapes fruit real photo",
            "purple grapes fruit real photo",
            "green grapes fruit real photo",
            "grapes fruit supermarket real photo",
            "葡萄 水果 实物照片",
            "紫葡萄 水果 实拍",
            "青葡萄 水果 实拍",
            "葡萄 水果 超市 实拍"
        ],
        "external_keywords": [
            "grapes fruit natural light photo",
            "grapes fruit market real photo",
            "grapes fruit kitchen photo",
            "葡萄 水果 自然光 实拍",
            "葡萄 水果 市场 实拍"
        ]
    },

    "strawberry": {
        "cn": "草莓",
        "raw_keywords": [
            "real strawberry fruit photo on table",
            "fresh strawberry fruit real photo",
            "strawberries fruit supermarket real photo",
            "strawberry fruit natural light photo",
            "草莓 水果 实物照片",
            "草莓 水果 桌面 实拍",
            "草莓 水果 超市 实拍",
            "新鲜草莓 实拍"
        ],
        "external_keywords": [
            "strawberry fruit market real photo",
            "strawberries fruit kitchen photo",
            "fresh strawberries natural light photo",
            "草莓 水果 自然光 实拍",
            "草莓 水果 市场 实拍"
        ]
    }
}


# =========================
# 工具函数
# =========================

def make_dirs():
    for base_dir in [RAW_DIR, EXTERNAL_TEST_DIR]:
        for class_name in CLASSES.keys():
            os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)


def count_images(folder):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    if not os.path.exists(folder):
        return 0

    return len([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ])


def is_valid_image(img_path, min_size=160):
    """
    检查图片是否能正常打开，并过滤尺寸太小的图片。
    """
    try:
        with Image.open(img_path) as img:
            img.verify()

        with Image.open(img_path) as img:
            w, h = img.size

        if w < min_size or h < min_size:
            return False

        return True

    except Exception:
        return False


def clean_invalid_images(folder):
    """
    删除损坏图片、太小图片、非图片文件。
    """
    files = os.listdir(folder)

    for file in files:
        path = os.path.join(folder, file)

        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(file)[1].lower()

        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            try:
                os.remove(path)
            except Exception:
                pass
            continue

        if not is_valid_image(path):
            try:
                os.remove(path)
            except Exception:
                pass


def convert_and_rename_images(folder, class_name, source_name):
    """
    统一转换为 jpg，并重命名。
    例如 apple_web_001.jpg
    """
    files = sorted([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    ])

    temp_dir = os.path.join(folder, "_temp_rename")
    os.makedirs(temp_dir, exist_ok=True)

    idx = 1

    for file in files:
        old_path = os.path.join(folder, file)

        if not os.path.isfile(old_path):
            continue

        ext = os.path.splitext(file)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue

        new_name = f"{class_name}_{source_name}_{idx:03d}.jpg"
        new_path = os.path.join(temp_dir, new_name)

        try:
            with Image.open(old_path) as img:
                img = img.convert("RGB")
                img.save(new_path, "JPEG", quality=95)
            idx += 1
        except Exception:
            pass

    # 删除原图片
    for file in files:
        old_path = os.path.join(folder, file)
        if os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except Exception:
                pass

    # 移回重命名图片
    for file in os.listdir(temp_dir):
        os.rename(os.path.join(temp_dir, file), os.path.join(folder, file))

    os.rmdir(temp_dir)


def trim_extra_images(folder, keep_num):
    """
    如果下载超过目标数量，只保留前 keep_num 张。
    """
    files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    ])

    if len(files) <= keep_num:
        return

    for file in files[keep_num:]:
        try:
            os.remove(os.path.join(folder, file))
        except Exception:
            pass


def download_by_keywords(keywords, save_dir, target_num):
    """
    按多个关键词下载图片，直到达到目标数量。
    """
    os.makedirs(save_dir, exist_ok=True)

    current_num = count_images(save_dir)

    if current_num >= target_num:
        print(f"已存在 {current_num} 张图片，跳过下载。")
        return

    round_id = 1

    while count_images(save_dir) < target_num:
        print(f"\n当前数量：{count_images(save_dir)} / {target_num}")
        print(f"开始第 {round_id} 轮下载...")

        for keyword in keywords:
            current_num = count_images(save_dir)
            if current_num >= target_num:
                break

            need_num = target_num - current_num

            # 每次关键词多下载一点，防止无效图片被清理后数量不够
            download_num = min(max(need_num + 10, 20), 60)

            print(f"关键词：{keyword}")
            print(f"计划下载：{download_num} 张")

            try:
                crawler = BingImageCrawler(storage={"root_dir": save_dir})
                crawler.crawl(
                    keyword=keyword,
                    max_num=download_num,
                    file_idx_offset="auto"
                )
            except Exception as e:
                print(f"下载失败，跳过该关键词：{keyword}")
                print("错误信息：", e)

            time.sleep(1)
            clean_invalid_images(save_dir)

            print(f"清理后数量：{count_images(save_dir)} / {target_num}")

        round_id += 1

        # 防止极端情况下死循环
        if round_id > 5:
            print("已尝试 5 轮下载，停止该类别下载。")
            break


def download_dataset(base_dir, num_per_class, keyword_type, source_name):
    """
    下载指定数据集。
    """
    for class_name, info in CLASSES.items():
        save_dir = os.path.join(base_dir, class_name)

        print("\n" + "=" * 70)
        print(f"类别：{class_name} / {info['cn']}")
        print(f"目标数量：{num_per_class}")
        print("=" * 70)

        keywords = info[keyword_type]

        download_by_keywords(
            keywords=keywords,
            save_dir=save_dir,
            target_num=num_per_class
        )

        clean_invalid_images(save_dir)
        trim_extra_images(save_dir, num_per_class)
        convert_and_rename_images(save_dir, class_name, source_name)

        final_num = count_images(save_dir)

        print(f"\n类别 {class_name} 下载完成，最终数量：{final_num}")

        if final_num < num_per_class:
            print(f"警告：{class_name} 数量不足，建议后续手动补充 {num_per_class - final_num} 张。")


def main():
    make_dirs()

    print("\n" + "#" * 80)
    print("开始下载主数据集 raw")
    print(f"每类目标数量：{RAW_NUM_PER_CLASS}")
    print("#" * 80)

    download_dataset(
        base_dir=RAW_DIR,
        num_per_class=RAW_NUM_PER_CLASS,
        keyword_type="raw_keywords",
        source_name="web"
    )

    print("\n" + "#" * 80)
    print("开始下载外部测试集 external_test")
    print(f"每类目标数量：{EXTERNAL_NUM_PER_CLASS}")
    print("#" * 80)

    download_dataset(
        base_dir=EXTERNAL_TEST_DIR,
        num_per_class=EXTERNAL_NUM_PER_CLASS,
        keyword_type="external_keywords",
        source_name="external_web"
    )

    print("\n全部下载完成！")
    print(f"主数据集路径：{RAW_DIR}")
    print(f"外部测试集路径：{EXTERNAL_TEST_DIR}")

    print("\n接下来请你人工检查每个类别文件夹，删除：")
    print("1. 不是该类别水果的图片")
    print("2. 卡通图、图标、插画")
    print("3. 水果主体太小、太模糊、太暗的图片")
    print("4. 多种水果混在一起且主体不明确的图片")
    print("5. 明显重复的图片")
    print("\n建议最终保留：")
    print("raw 每类至少 100 张")
    print("external_test 每类至少 20 张")


if __name__ == "__main__":
    main()