
import os
import json
import random
from collections import defaultdict

# ================= 配置区域 =================
PROJECT_ROOT = "/root/sj-tmp/Med-CLIP-SAM"
DATA_PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "data", "processed")

# 显式2D/3D，
DATASET_2D = [
    "BUSI_256_Expert_PreserveAR",
    "shenzhen_ChestXray_256_Expert_Std",
    "DRIVE_256_Expert_FOV",
    "ISIC_256_Expert_DullRazor",
    "Kvasir_256_Expert_SpecRemoval",
    "Nerve_256_Expert_Std",
    "SIIM_256_Expert_Std",
    "CVC_ClinicDB_256_Expert_Std",
    "TN3K_256_Expert_Std",
    "ETIS_Larib_256_Expert_Std"
    
]
DATASET_3D = [
    "ACDC_256_2.5D_Expert",
    "BraTS_256_MultiModal_RGB_v1",
    "CAMUS_256_Expert_Std",
    "CHAOS_CT_256_Expert",
    "CHAOS_MRI_256_Expert_2.5D_Simple",
    "KiTS19_256_Expert_2D",
    "LiTS_256_Expert",
    "TotalSegmentator_256_Balanced_2.5D_Expert",
    "MSD_Task02_256_2.5D_Expert",
    "MSD_Task04_256_Expert_SR",
    "MSD_Task05_256_Expert_Std",
    "MSD_Task06_256_Expert_2.5D",
    
    
]
DATASET_NAMES = DATASET_2D + DATASET_3D

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 38


# ================= 工具函数 =================
def fix_path_format(path_str, correct_dataset_name):
    path_str = path_str.replace("\\", "/")
    parts = path_str.split("/")
    new_parts = []
    for part in parts:
        if part.lower() == correct_dataset_name.lower():
            new_parts.append(correct_dataset_name)
        else:
            new_parts.append(part)
    return "/".join(new_parts)


def get_patient_id(sample_entry, dataset_name):
    img_path = sample_entry['img_path']
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')

    # 3D 按患者提取ID
    if dataset_name in DATASET_3D:
        if "CHAOS_CT" in dataset_name:
            return "_".join(parts[:-1])
        elif "CHAOS_MRI" in dataset_name:
            return "_".join(parts[:3]) if len(parts) >= 3 else "_".join(parts[:-1])
        elif "BraTS" in dataset_name:
            return "_".join(parts[:-1])
        elif "TotalSegmentator" in dataset_name:
            return parts[0]
        elif "ACDC" in dataset_name:
            for part in parts:
                if "patient" in part:
                    return f"acdc_{part}"
            return "_".join(parts[:-1])
        elif "KiTS19" in dataset_name:
            return "_".join(parts[:3]) if len(parts) >= 3 else "_".join(parts[:-1])
        elif "LiTS" in dataset_name:
            return "_".join(parts[:-1])
        elif "CAMUS" in dataset_name:
            if len(parts) >= 2 and "patient" in parts[1]:
                return f"{parts[0]}_{parts[1]}"
            return parts[0] if "patient" in parts[0] else "_".join(parts[:-1])
        elif "MSD" in dataset_name:
            return "_".join(parts[:-1])
       
    # 2D 直接用唯一文件名，1个ID=1张图
    return f"{dataset_name}_{base_name}"


# ================= 修复划分函数 =================
def split_3d_patient_group(patient_groups):
    """3D专用：按患者组划分，严格8:1:1，无越界无重复"""
    pids = list(patient_groups.keys())
    random.shuffle(pids)
    n_total = len(pids)

    # 核心修复：计算+上限锁定
    n_train = min(int(n_total * TRAIN_RATIO), n_total)
    n_val = min(int(n_total * VAL_RATIO), n_total - n_train)
    n_test = n_total - n_train - n_val

    train_ids = pids[:n_train]
    val_ids = pids[n_train:n_train + n_val]
    test_ids = pids[n_train + n_val:]

    train_set = [s for pid in train_ids for s in patient_groups[pid]]
    val_set = [s for pid in val_ids for s in patient_groups[pid]]
    test_set = [s for pid in test_ids for s in patient_groups[pid]]

    return train_set, val_set, test_set, n_total


def split_2d_direct(sample_list):
    """2D专用：直接打乱切分，绝无比例错误/重复"""
    random.shuffle(sample_list)
    n_total = len(sample_list)

    n_train = min(int(n_total * TRAIN_RATIO), n_total)
    n_val = min(int(n_total * VAL_RATIO), n_total - n_train)

    train_set = sample_list[:n_train]
    val_set = sample_list[n_train:n_train + n_val]
    test_set = sample_list[n_train + n_val:]

    return train_set, val_set, test_set, n_total


# ================= 主函数 =================
def main():
    random.seed(RANDOM_SEED)
    final_train, final_val, final_test = [], [], []

    print(f"开始执行")
    print(f"{'Dataset Folder':<30} | {'Type':<4} | {'Total':<6} | {'Train':<6} | {'Val':<5} | {'Test':<5}")
    print("-" * 85)

    for ds_name in DATASET_NAMES:
        json_path = os.path.join(DATA_PROCESSED_ROOT, ds_name, "dataset_metadata.json")
        if not os.path.exists(json_path):
            print(f"⚠️ 跳过: {ds_name} (文件不存在)")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data:
            print(f"⚠️ 跳过: {ds_name} (空数据)")
            continue

        # 路径格式化
        for sample in data:
            sample['img_path'] = fix_path_format(sample['img_path'], ds_name)
            sample['mask_path'] = fix_path_format(sample['mask_path'], ds_name)

        # 2D / 3D 分流划分
        if ds_name in DATASET_2D:
            train, val, test, total = split_2d_direct(data)
            type_str = "2D"
        else:
            groups = defaultdict(list)
            for sample in data:
                pid = get_patient_id(sample, ds_name)
                groups[pid].append(sample)
            train, val, test, total = split_3d_patient_group(groups)
            type_str = "3D"

        # 汇总
        final_train.extend(train)
        final_val.extend(val)
        final_test.extend(test)

        # 打印（严格校验和=总数）
        print(f"{ds_name:<30} | {type_str:<4} | {len(data):<6} | {len(train):<6} | {len(val):<5} | {len(test):<5}")

    print("-" * 85)

    # 保存
    os.makedirs(DATA_PROCESSED_ROOT, exist_ok=True)
    with open(os.path.join(DATA_PROCESSED_ROOT, "total_train.json"), "w", encoding='utf-8') as f:
        json.dump(final_train, f, indent=4)
    with open(os.path.join(DATA_PROCESSED_ROOT, "total_val.json"), "w", encoding='utf-8') as f:
        json.dump(final_val, f, indent=4)
    with open(os.path.join(DATA_PROCESSED_ROOT, "total_test.json"), "w", encoding='utf-8') as f:
        json.dump(final_test, f, indent=4)

   
    print(f"Train: {len(final_train)} | Val: {len(final_val)} | Test: {len(final_test)}")
    print(f"总和校验: {len(final_train)+len(final_val)+len(final_test)}")


if __name__ == "__main__":
    main()