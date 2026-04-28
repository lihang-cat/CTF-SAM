import os
import glob
import json
import numpy as np
import nibabel as nib
import cv2
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\MSD\Task10_Colon"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MSD_Task10_256_Expert_2.5D"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# 🌟 最新研究趋势：2.5D 空间堆叠，赋予 2D 视觉模型感知肠道连续走向的能力
USE_25D = True

# ================= 2. Definitions =================

# Label 1: Colon Tumor (结肠肿瘤)
TASK10_ORGAN_MAP = {
    1: "colon tumor"
}

# Expert Description (统一标准字典格式)
# 强调 "bowel wall thickening" (肠壁增厚) 有助于文本编码器聚焦特征
TASK10_EXPERT_DESC = {
    "colon tumor": "colon tumor, a bright irregular thickening of the colon wall"
}


# ================= 3. Core Functions =================

def normalize_ct_window(img_slice, window_min, window_max):
    """ CT 专用归一化: 根据 HU 值窗口进行截断 """
    img_slice = np.clip(img_slice, window_min, window_max)
    if window_max == window_min: return np.zeros_like(img_slice, dtype=np.uint8)
    img_slice = (img_slice - window_min) / (window_max - window_min)
    return (img_slice * 255).astype(np.uint8)


def resize_pad(image, target_size, is_mask=False):
    """ 🌟 OpenCV 底层极速 Padding """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    padding_value = 0
    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def process_single_case(args):
    """ 多进程 Worker """
    img_path, mask_path, subject_id, output_img_dir, output_mask_dir = args
    meta_list = []

    try:
        # 🌟 规范化解剖学方向 (RAS)，保证文本大模型空间感知的一致性
        img_nii = nib.as_closest_canonical(nib.load(img_path))
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))

        # 强制浮点/整型读取 dataobj 以降低 OOM 风险，立即释放 nibabel 句柄
        img_data = np.asanyarray(img_nii.dataobj, dtype=np.float32)
        mask_data = np.asanyarray(mask_nii.dataobj, dtype=np.uint8)
        del img_nii, mask_nii

        if img_data.ndim == 4: img_data = img_data[:, :, :, 0]
        if mask_data.ndim == 4: mask_data = mask_data[:, :, :, 0]

        if img_data.shape != mask_data.shape: return []

        depth = img_data.shape[2]

        # 快速跳过无目标的空白切片
        has_target_z = np.any(mask_data > 0, axis=(0, 1))
        valid_slices = np.where(has_target_z)[0]

        # 结肠肿瘤观察窗：标准软组织偏窄窗，滤除脂肪和游离气体，聚焦实体肿块
        W_MIN, W_MAX = -100, 200

        for z in valid_slices:
            mask_slice = mask_data[:, :, z]
            if np.sum(mask_slice) == 0: continue

            # === Image Processing ===
            if USE_25D:
                # 🌟 2.5D 空间堆叠：将肠管在上下层的连续性输入 RGB 通道
                idx_prev = max(0, z - 1)
                idx_next = min(depth - 1, z + 1)

                s_prev = normalize_ct_window(img_data[:, :, idx_prev], W_MIN, W_MAX)
                s_curr = normalize_ct_window(img_data[:, :, z], W_MIN, W_MAX)
                s_next = normalize_ct_window(img_data[:, :, idx_next], W_MIN, W_MAX)

                img_rgb_slice = np.stack([s_prev, s_curr, s_next], axis=-1)
                modality_str = "CT"
            else:
                # 原始单层多窗口融合策略 (后备方案)
                img_slice = img_data[:, :, z]
                ch_r = normalize_ct_window(img_slice, -100, 200)  # 标准软组织
                ch_g = normalize_ct_window(img_slice, 20, 150)  # 强化肿块对比度
                ch_b = normalize_ct_window(img_slice, -1000, 1000)  # 宽窗看肠道气体结构

                img_rgb_slice = np.stack([ch_r, ch_g, ch_b], axis=-1)
                modality_str = "CT"

            # === Resize & Pad ===
            img_padded = resize_pad(img_rgb_slice, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

            file_base_id = f"msd_task10_{subject_id}_{z:03d}"

            # 复用机制，同一层图片仅保存一次
            img_saved = False
            temp_meta = []

            # Process Organs
            unique_vals = np.unique(mask_padded)
            for val in unique_vals:
                if val == 0 or val not in TASK10_ORGAN_MAP: continue

                organ_name = TASK10_ORGAN_MAP[val]
                binary_mask = (mask_padded == val).astype(np.uint8)

                # 🌟 必须在 Resize 后进行过滤，防止生成缩放导致消失的空 Mask 脏数据
                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD:
                    continue

                # 确认有有效掩码后，再保存原图 (Only Once)
                if not img_saved:
                    img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
                    np.save(img_save_path, img_padded)
                    rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
                    img_saved = True

                # Save Mask
                task_file_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{task_file_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                desc_text = TASK10_EXPERT_DESC[organ_name]

                # ✅ Standardized Metadata Format (丢弃旧随机 Prompt)
                temp_meta.append({
                    "img_path": rel_img_path,
                    "mask_path": rel_mask_path,
                    "modality": modality_str,
                    "description": desc_text,
                    "organ": organ_name,
                    "raw_organ_id": int(val),
                    "source": "MSD_Task10"
                })

            if img_saved:
                meta_list.extend(temp_meta)

    except Exception as e:
        print(f"\n❌ Error processing {subject_id}: {str(e)}")
        print(traceback.format_exc())
        return []

    return meta_list


# ================= 4. Main =================

def main():
    multiprocessing.freeze_support()

    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting MSD Task10 (Colon) Preprocessing")
    print(f"✨ Strategy: {'Spatial 2.5D Stacking' if USE_25D else '2D Multi-Window'} | Soft Tissue Focus")
    print(f"⚡ Workers: {NUM_WORKERS}")

    # Find Folders
    img_folder = os.path.join(DATASET_ROOT, "imagesTr")
    lbl_folder = os.path.join(DATASET_ROOT, "labelsTr")

    if not os.path.exists(img_folder):
        for root, dirs, files in os.walk(DATASET_ROOT):
            if "imagesTr" in dirs:
                img_folder = os.path.join(root, "imagesTr")
                lbl_folder = os.path.join(root, "labelsTr")
                break

    if not os.path.exists(img_folder):
        print("❌ 找不到 imagesTr")
        return

    # Build Tasks (过滤系统隐藏文件)
    tasks = []
    img_files = [f for f in glob.glob(os.path.join(img_folder, "*.nii*")) if not os.path.basename(f).startswith("._")]

    for img_path in img_files:
        filename = os.path.basename(img_path)
        mask_path = os.path.join(lbl_folder, filename)
        if os.path.exists(mask_path):
            subject_id = filename.split(".")[0]
            tasks.append((img_path, mask_path, subject_id, img_dir, mask_dir))

    print(f"✅ Found {len(tasks)} cases, ready to process...")

    all_metadata = []

    # 榨干 CPU 多核性能并发处理
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_case, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Cases"):
            res = future.result()
            if res: all_metadata.extend(res)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print("-" * 40)
    print(f"🎉 Finished! Successful Samples: {len(all_metadata)}")
    print(f"📂 Metadata saved: {json_path}")


if __name__ == "__main__":
    main()