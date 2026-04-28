import os
import glob
import random
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 配置区域 =================

# 🔴 路径配置
DATASET_ROOT = r"D:\dataset\ACDC"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\ACDC_256_2.5D_Expert"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# 🔴 参数配置
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20  # 过滤微小噪点
USE_25D = True  # 🌟 开启 2.5D 堆叠 (强烈推荐)
NUM_WORKERS = 8  # ⚡ 进程数 (根据 CPU 核心数调整)

# ================= 2. ACDC 专属定义 =================

ACDC_ORGAN_MAP = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle"
}

# 专家级解剖描述 (增强文本引导特征对齐)
ACDC_EXPERT_DESC = {
    1: "right ventricle cavity, the chamber that pumps deoxygenated blood to the lungs",
    2: "myocardium, the thick muscle layer of the left ventricle wall",
    3: "left ventricle cavity, the main pumping chamber of the heart"
}


# ================= 3. 核心工具函数 =================

def read_acdc_info(cfg_path):
    """读取 Info.cfg 获取 ED 和 ES 的帧序号"""
    info = {}
    if not os.path.exists(cfg_path): return None, None
    with open(cfg_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                info[parts[0].strip()] = parts[1].strip()
    try:
        return int(info.get('ED', -1)), int(info.get('ES', -1))
    except ValueError:
        return None, None


def normalize_volume_3d(img_volume):
    """
    🌟 [核心优化] 3D 级一致性归一化
    保持心脏在整个 3D 扫描中的亮度一致性，优于单切片归一化
    """
    mask = img_volume > 0
    if np.sum(mask) == 0: return img_volume

    pixels = img_volume[mask]
    # 0.5% - 99.5% 鲁棒截断，去除 MRI 高亮伪影
    lower = np.percentile(pixels, 0.5)
    upper = np.percentile(pixels, 99.5)

    img_volume = np.clip(img_volume, lower, upper)

    if upper == lower: return np.zeros_like(img_volume)

    # 归一化到 0-1 (Float32)
    img_volume = (img_volume - lower) / (upper - lower)
    return img_volume


def apply_2d_enhancement(img_slice_norm):
    """ 将 0-1 float 转为 uint8 并应用 CLAHE 增强对比度 """
    img_uint8 = (img_slice_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def get_25d_slice(volume, z_index):
    """
    🌟 [核心优化] 获取 2.5D 切片 [z-1, z, z+1]
    利用相邻切片提供空间上下文，解决心脏基底部/心尖部模糊问题
    """
    depth = volume.shape[2]

    # 当前层
    slice_curr = volume[:, :, z_index]

    # 上下层 (边界处复制当前层)
    idx_prev = max(0, z_index - 1)
    slice_prev = volume[:, :, idx_prev]

    idx_next = min(depth - 1, z_index + 1)
    slice_next = volume[:, :, idx_next]

    # 分别增强
    img_prev = apply_2d_enhancement(slice_prev)
    img_curr = apply_2d_enhancement(slice_curr)
    img_next = apply_2d_enhancement(slice_next)

    # 堆叠为 RGB (H, W, 3)
    return np.stack([img_prev, img_curr, img_next], axis=-1)


def resize_pad(image, target_size, is_mask=False):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 图像用线性插值，Mask 必须用最近邻
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w

    # 居中 Padding
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant')
    else:
        # 3通道图像 Padding
        if not is_mask:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant')


# ================= 4. 单例处理逻辑 (支持多进程) =================

def process_single_patient(args):
    folder_path, subject_id, output_img_dir, output_mask_dir = args
    cfg_path = os.path.join(folder_path, "Info.cfg")

    # 读取 ED/ES 帧号
    ed_frame, es_frame = read_acdc_info(cfg_path)
    if ed_frame is None: return []

    case_meta_list = []

    # 需要处理的两个时相
    targets = [("ED", ed_frame), ("ES", es_frame)]

    for phase_name, frame_num in targets:
        frame_str = f"{frame_num:02d}"
        base_name = f"{subject_id}_frame{frame_str}"

        # 兼容 .nii 和 .nii.gz
        img_path = os.path.join(folder_path, f"{base_name}.nii.gz")
        gt_path = os.path.join(folder_path, f"{base_name}_gt.nii.gz")
        if not os.path.exists(img_path):
            img_path = os.path.join(folder_path, f"{base_name}.nii")
            gt_path = os.path.join(folder_path, f"{base_name}_gt.nii")

        if not (os.path.exists(img_path) and os.path.exists(gt_path)):
            continue

        try:
            # 1. 统一方向为 RAS (Canonical)
            img_nii = nib.as_closest_canonical(nib.load(img_path))
            mask_nii = nib.as_closest_canonical(nib.load(gt_path))

            img_data = img_nii.get_fdata().astype(np.float32)
            mask_data = mask_nii.get_fdata().astype(np.uint8)

            # 移除多余维度
            if img_data.ndim == 4: img_data = img_data[:, :, :, 0]
            if mask_data.ndim == 4: mask_data = mask_data[:, :, :, 0]

            # 2. 3D 归一化 (关键步骤)
            img_data_norm = normalize_volume_3d(img_data)
            depth = img_data.shape[2]

            for z in range(depth):
                mask_slice = mask_data[:, :, z]

                # 跳过无标签切片
                if np.sum(mask_slice) == 0: continue

                # === 2.5D 切片提取 ===
                if USE_25D:
                    img_final = get_25d_slice(img_data_norm, z)
                else:
                    processed = apply_2d_enhancement(img_data_norm[:, :, z])
                    img_final = np.stack([processed] * 3, axis=-1)

                # Resize & Pad
                img_padded = resize_pad(img_final, IMG_SIZE, is_mask=False)
                mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

                file_base_id = f"acdc_{subject_id}_{phase_name}_{z:02d}"

                # 保存图像
                img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
                np.save(img_save_path, img_padded)
                rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

                # === 标签分离 ===
                unique_vals = np.unique(mask_padded)
                for val in unique_vals:
                    if val == 0 or val not in ACDC_ORGAN_MAP: continue

                    binary_mask = (mask_padded == val).astype(np.uint8)

                    # 噪点过滤
                    if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

                    organ_name = ACDC_ORGAN_MAP[val]
                    desc_text = ACDC_EXPERT_DESC.get(val, organ_name)

                    task_file_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                    mask_save_path = os.path.join(output_mask_dir, f"{task_file_id}.npy")
                    np.save(mask_save_path, binary_mask)
                    rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                    # ✅ 构造精简的元数据 (适配新 Dataset)
                    case_meta_list.append({
                        "img_path": rel_img_path,
                        "mask_path": rel_mask_path,
                        "modality": "MRI",  # 🌟 属性：模态
                        "description": desc_text,  # 🌟 属性：详细描述
                        "organ": organ_name,  # 🌟 属性：器官名
                        "phase": phase_name,  # 额外信息：ED/ES
                        "raw_organ_id": int(val),
                        "source": "ACDC"
                    })

        except Exception as e:
            print(f"❌ Error in {subject_id}: {e}")

    return case_meta_list


# ================= 5. 主程序 =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 开始 ACDC 多进程预处理")
    print(f"⚙️  分辨率: {IMG_SIZE} | 2.5D模式: {USE_25D} | 进程数: {NUM_WORKERS}")

    # 收集任务
    tasks = []
    # ACDC 数据集结构通常是 database/training/patientXXX
    # 为了兼容性，搜索所有 patient* 文件夹
    search_paths = [
        os.path.join(DATASET_ROOT, "database", "training"),
        os.path.join(DATASET_ROOT, "database", "testing"),
        DATASET_ROOT  # 兼容解压在根目录的情况
    ]

    seen_patients = set()
    for root_path in search_paths:
        if not os.path.exists(root_path): continue
        patient_folders = glob.glob(os.path.join(root_path, "patient*"))

        for folder in patient_folders:
            subject_id = os.path.basename(folder)
            if subject_id in seen_patients: continue  # 避免重复
            seen_patients.add(subject_id)

            tasks.append((folder, subject_id, img_dir, mask_dir))

    print(f"✅ 发现 {len(tasks)} 个病人数据")

    all_meta_data = []

    # ⚡ 启动多进程
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_patient, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Patients"):
            result = future.result()
            if result:
                all_meta_data.extend(result)

    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=4)

    print("-" * 50)
    print(f"🎉 处理完成！总样本数: {len(all_meta_data)}")
    print(f"📂 JSON 已保存至: {json_path}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 路径不存在: {DATASET_ROOT}")
    else:
        main()