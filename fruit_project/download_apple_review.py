import os
from icrawler.builtin import BingImageCrawler

SAVE_DIR = "data/raw/apple"

os.makedirs(SAVE_DIR, exist_ok=True)

keywords = [
    "yellow apple fruit",
    "green apple fruit",
    "golden apple fruit",
    "yellow green apple fruit",
    "small apples fruit",
    "multiple apples fruit",
    "apples in basket",
    "apples on tree",
    "red apples cluster",
    "apple bunch fruit",
    "single apple with stem",
    "apple with leaf fruit",
]

# 每个关键词下载多少张
max_num_per_keyword = 15

for keyword in keywords:
    print(f"Downloading: {keyword}")

    crawler = BingImageCrawler(
        storage={"root_dir": SAVE_DIR}
    )

    crawler.crawl(
        keyword=keyword,
        max_num=max_num_per_keyword,
        min_size=(200, 200),
        file_idx_offset="auto"
    )

print("Done.")
print(f"Images saved to: {SAVE_DIR}")