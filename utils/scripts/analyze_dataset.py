import json
import os
from collections import Counter
import pandas as pd

# 你的 metadata 路径
JSON_PATH = (r"E:\code\Med-CLIP-SAM\data\processed\BraTS_256_MultiModal_RGB\dataset_metadata.json")


def analyze_data():
    if not os.path.exists(JSON_PATH):
        print("❌ 找不到 metadata.json，请检查路径。")
        return

    print(f"正在读取 {JSON_PATH} ...")
    with open(JSON_PATH, 'r') as f:
        data_list = json.load(f)

    # 1. 核心统计：总切片数
    total_slices = len(data_list)
    print("=" * 40)
    print(f"🟢 总切片数量 (Total Slices): {total_slices}")
    print("=" * 40)

    # 2. 进阶统计：每个器官有多少张？
    # 提取所有样本里的 'organ' 字段
    organs = [item['organ'] for item in data_list]
    organ_counts = Counter(organs)

    # 3. 转为 DataFrame 方便查看
    df = pd.DataFrame.from_dict(organ_counts, orient='index', columns=['count'])
    df = df.sort_values(by='count', ascending=False)

    print(f"📊 包含器官种类数: {len(df)}")
    print("-" * 40)

    # ========== 新增：打印所有器官类别及数量 ==========
    print("所有器官类别及数量 (All Organ Categories and Counts):")
    # 格式化输出，让每个器官和数量对齐，更易读
    for organ, count in df.itertuples():
        print(f"  {organ:<25} {count:>6}")
    print("-" * 40)
    # ================================================

    print("排名前 10 的器官 (Top 10 Organs):")
    print(df.head(10))
    print("-" * 40)
    print("排名后 5 的器官 (Bottom 5 Organs):")
    print(df.tail(5))
    print("-" * 40)

    # 4. 检查是否有数据量过少的器官 (风险提示)
    low_data_organs = df[df['count'] < 50]
    if not low_data_organs.empty:
        print(f"⚠️ 警告: 有 {len(low_data_organs)} 个器官样本数少于 50 张，可能会导致过拟合。")
        print(low_data_organs.index.tolist())
    else:
        print("✅ 数据平衡性检查通过，所有器官样本充足。")


if __name__ == "__main__":
    analyze_data()