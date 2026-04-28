import os
import glob
import re
import json
import numpy as np
import nibabel as nib
import cv2
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. Configuration =================

# 你的 LiTS 数据集根目录 (包含 volume-xx.nii 和 segmentation-xx.nii)
DATASET_ROOT = r"D:\dataset\LiTS"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\LiTS_256_Expert"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 50  # 稍微调高一点，忽略太小的噪点
NUM_WORKERS = 4  # ⚠️ NIfTI 很吃内存，建议设为 CPU 核心数的一半
USE_25D = False  # 🌟 True=[z-1, z, z+1], False=[LiverWindow]*3

# ================= 2. Definitions =================

LITS_ORGAN_MAP = {
    1: "liver",
    2: "liver tumor"
}

LITS_EXPERT_DESC = {
    "liver": "liver, a large vascular organ in the upper right abdomen",
    "liver tumor": "liver tumor, a focal lesion within the liver parenchyma, often hypoattenuating"
}


# ================= 3. Core Functions =================

def apply_window(img, center, width):
    """ Apply Liver Window (Center=30, Width=150) """
    lower = center - width / 2.0
    upper = center + width / 2.0
    img_clamped = np.clip(img, lower, upper)
    # 归一化到 [0, 1]
    return (img_clamped - lower) / (upper - lower) if width > 0 else np.zeros_like(img)


def get_25d_slice(vol_data, z_index):
    """
    Strategy: Spatial 2.5D with Context [z-1, z, z+1]
    """
    depth = vol_data.shape[2]  # NIfTI 默认形状是 (H, W, D)

    # Liver Window Settings
    W_CENTER, W_WIDTH = 30, 150

    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    # 提取切片
    slice_prev = apply_window(vol_data[:, :, idx_prev], W_CENTER, W_WIDTH)
    slice_curr = apply_window(vol_data[:, :, idx_curr], W_CENTER, W_WIDTH)
    slice_next = apply_window(vol_data[:, :, idx_next], W_CENTER, W_WIDTH)

    # Stack -> (H, W, 3) -> uint8
    img_merged = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
    return (img_merged * 255).astype(np.uint8)


def resize_pad(image, target_size, is_mask=False):
    """
    Resize image to fit within target_size while keeping aspect ratio, then pad.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 插值方式：Mask用最近邻，图像用线性插值
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 计算 Padding
    delta_h = target_size - new_h
    delta_w = target_size - new_w
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padding_val = 0  # 黑色填充

    if is_mask:
        # Mask 是 (H, W)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_val)
    else:
        # Image 是 (H, W, 3)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def process_single_volume(args):
    """ Worker Function """
    vol_path, mask_path, subject_id, output_img_dir, output_mask_dir = args
    meta_list = []

    try:
        # 1. 内存安全加载 (Memory Safe Load)
        # 使用 mmap 模式，且不直接转 float64，转 float32 节省一半内存
        ct_nii = nib.load(vol_path)
        mask_nii = nib.load(mask_path)

        # 强制修正方向 (Canonical RAS)，保证 CT 和 Mask 方向一致
        # 注意：这一步会触发读取，比较耗时，但为了对齐是必须的
        ct_nii = nib.as_closest_canonical(ct_nii)
        mask_nii = nib.as_closest_canonical(mask_nii)

        ct_vol = np.asanyarray(ct_nii.dataobj, dtype=np.float32)
        mask_vol = np.asanyarray(mask_nii.dataobj, dtype=np.uint8)

        if ct_vol.shape != mask_vol.shape:
            print(f"⚠️ Shape mismatch for {subject_id}: {ct_vol.shape} vs {mask_vol.shape}")
            return []

        # 2. 筛选有效层 (只处理有肝脏或肿瘤的层)
        # axis=(0, 1) 表示计算每一层的 sum，得到形状 (D,)
        slice_sums = np.sum(mask_vol > 0, axis=(0, 1))
        valid_slices = np.where(slice_sums > 0)[0]

        # 降采样：如果层数太多，可以每隔一层取一个 (可选)
        # valid_slices = valid_slices[::2]

        for idx, z in enumerate(valid_slices):
            # === Image Processing ===
            if USE_25D:
                img_processed = get_25d_slice(ct_vol, z)
                modality_str = "CT (Spatial 2.5D)"
            else:
                # 2D 模式
                slice_norm = apply_window(ct_vol[:, :, z], 30, 150)
                slice_uint8 = (slice_norm * 255).astype(np.uint8)
                img_processed = np.stack([slice_uint8] * 3, axis=-1)
                modality_str = "CT"

            # === Debug: 保存第一张图用于人工检查 ===
            # (只在第一个处理的病人的中间层触发一次)
            if subject_id == "0" and idx == len(valid_slices) // 2:
                debug_path = os.path.join(PROJECT_ROOT, "debug_check_lits.jpg")
                mask_vis = (mask_vol[:, :, z] * 120).astype(np.uint8)  # 简单可视化
                img_vis = img_processed[:, :, 1]  # 取中间通道
                overlay = cv2.addWeighted(img_vis, 0.7, mask_vis, 0.3, 0)
                cv2.imwrite(debug_path, overlay)
                print(f"🔎 [DEBUG] Overlay saved to: {debug_path}")

            # === Resize & Pad ===
            img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
            mask_slice = mask_vol[:, :, z]
            mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

            # 文件名 ID
            file_base_id = f"lits_{subject_id}_{z:03d}"

            # 保存 Image (只需保存一次)
            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_padded)

            # 存相对路径用于 JSON
            rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            # === Mask Processing (多类别拆分) ===
            unique_vals = np.unique(mask_padded)

            for val in unique_vals:
                if val == 0 or val not in LITS_ORGAN_MAP:
                    continue

                # 生成二值 Mask
                binary_mask = (mask_padded == val).astype(np.uint8)

                # 再次过滤太小的目标 (比如 Resize 后消失的小肿瘤)
                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD:
                    continue

                organ_name = LITS_ORGAN_MAP[val]
                task_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"

                mask_save_path = os.path.join(output_mask_dir, f"{task_id}.npy")
                np.save(mask_save_path, binary_mask)

                rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                meta_list.append({
                    "img_path": rel_img_path,
                    "mask_path": rel_mask_path,
                    "modality": modality_str,
                    "description": LITS_EXPERT_DESC[organ_name],
                    "organ": organ_name,
                    "raw_organ_id": int(val),
                    "source": "LiTS"
                })

    except Exception as e:
        print(f"❌ Error processing Volume {subject_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

    return meta_list


# ================= 4. Main =================

def main():
    # Windows 下多进程必须加这句
    multiprocessing.freeze_support()

    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting LiTS Preprocessing")
    print(f"📂 Dataset Root: {DATASET_ROOT}")
    print(f"✨ Strategy: {'Spatial 2.5D' if USE_25D else '2D Standard'}")
    print(f"⚡ Workers: {NUM_WORKERS}")

    # === 智能文件搜索 ===
    # 递归查找所有 volume-*.nii
    # 兼容 LiTS 常见的 'Training Batch 1/volume-0.nii' 或直接在根目录
    print("🔍 Scanning for NIfTI files...")
    all_nii_files = glob.glob(os.path.join(DATASET_ROOT, "**", "volume-*.nii"), recursive=True)

    tasks = []
    for vol_path in all_nii_files:
        # 从文件名解析 ID: volume-13.nii -> 13
        filename = os.path.basename(vol_path)
        match = re.search(r"volume-(\d+)\.nii", filename)
        if not match: continue

        idx = match.group(1)
        parent_dir = os.path.dirname(vol_path)

        # 尝试在同级目录找 segmentation-xx.nii
        # 或者 LiTS 可能会把 segmentation 放在单独的文件夹，这里假设在同级或 seg 文件夹
        possible_mask_paths = [
            os.path.join(parent_dir, f"segmentation-{idx}.nii"),  # 同级
            os.path.join(DATASET_ROOT, "segmentations", f"segmentation-{idx}.nii"),  # 根目录 seg 文件夹
            os.path.join(DATASET_ROOT, "Training Batch 2", f"segmentation-{idx}.nii")  # Batch 2
        ]

        mask_path = None
        for p in possible_mask_paths:
            if os.path.exists(p):
                mask_path = p
                break

        if mask_path:
            tasks.append((vol_path, mask_path, idx, img_dir, mask_dir))
        else:
            print(f"⚠️ Warning: No mask found for volume-{idx}.nii")

    print(f"✅ Found {len(tasks)} pairs of Volume/Mask.")

    if len(tasks) == 0:
        print("❌ Error: No valid data found. Check DATASET_ROOT.")
        return

    all_metadata = []

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_volume, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Volumes"):
            result = future.result()
            if result:
                all_metadata.extend(result)

    # 保存元数据
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"🎉 Finished! Total samples generated: {len(all_metadata)}")
    print(f"📂 Metadata saved to: {json_path}")
    print(f"🖼️ Check 'debug_check_lits.jpg' in project root to verify mask alignment.")


if __name__ == "__main__":
    main()