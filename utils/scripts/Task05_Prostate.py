import os
import glob
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\MSD\Task05_Prostate"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MSD_Task05_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
TARGET_SPACING = (0.6, 0.6)  # 统一的物理平面分辨率 (mm/pixel)
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# ================= 2. Definitions =================

TASK05_ORGAN_MAP = {
    1: "peripheral zone",
    2: "transition zone"
}

# 专家级描述
TASK05_EXPERT_DESC = {
    "peripheral zone": "peripheral zone, the bright outer region of the prostate on T2-weighted MRI",
    "transition zone": "transition zone, the heterogeneous central region of the prostate"
}


# ================= 3. Core Functions =================

def get_global_bounds(volume):
    """
    提取 3D 体数据的全局分位数边界，保持 3D 灰度连续性
    """
    mask = volume > (np.mean(volume) * 0.1)
    if np.sum(mask) == 0:
        return 0, 1

    pixels = volume[mask]
    lower = np.percentile(pixels, 0.5)
    upper = np.percentile(pixels, 99.5)
    return lower, upper


def apply_normalization_with_bounds(img_slice, lower, upper):
    """
    使用全局边界进行局部切片的归一化 + CLAHE
    """
    img_slice = np.clip(img_slice, lower, upper)

    if upper == lower:
        return np.zeros_like(img_slice, dtype=np.uint8)

    img_norm = (img_slice - lower) / (upper - lower)
    img_uint8 = (img_norm * 255).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def resample_and_pad_crop(image, original_spacing, target_spacing, target_size, is_mask=False):
    """
    基于物理间距重采样 (Spacing Resampling)，防止器官形变，
    然后再进行 Center Crop 或 Pad 以匹配网络输入大小。
    """
    h, w = image.shape[:2]

    # 计算基于物理间距的缩放比例
    scale_y = original_spacing[0] / target_spacing[0]
    scale_x = original_spacing[1] / target_spacing[1]

    new_h, new_w = int(h * scale_y), int(w * scale_x)

    # 1. 物理重采样
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 2. Center Crop or Pad 到 target_size (256x256)
    final_img = np.zeros((target_size, target_size) + image.shape[2:], dtype=image.dtype)

    pad_y = max(0, target_size - new_h)
    pad_x = max(0, target_size - new_w)
    crop_y = max(0, new_h - target_size)
    crop_x = max(0, new_w - target_size)

    start_y_img = crop_y // 2
    start_x_img = crop_x // 2
    roi_img = resized[start_y_img:start_y_img + min(new_h, target_size),
    start_x_img:start_x_img + min(new_w, target_size)]

    start_y_final = pad_y // 2
    start_x_final = pad_x // 2
    final_img[start_y_final:start_y_final + roi_img.shape[0],
    start_x_final:start_x_final + roi_img.shape[1]] = roi_img

    return final_img


def process_single_case(args):
    """ Worker Function """
    img_path, mask_path, subject_id, output_img_dir, output_mask_dir = args
    meta_list = []

    try:
        # Load NIfTI
        img_nii = nib.as_closest_canonical(nib.load(img_path))
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))

        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.uint8)

        # 提取物理间距 (x, y, z)
        spacing = img_nii.header.get_zooms()[:2]

        if img_data.ndim != 4 or img_data.shape[3] < 2: return []
        if mask_data.ndim == 4: mask_data = mask_data[:, :, :, 0]

        depth = img_data.shape[2]

        # 获取 3D 全局归一化边界
        t2_volume, adc_volume = img_data[:, :, :, 0], img_data[:, :, :, 1]
        t2_lower, t2_upper = get_global_bounds(t2_volume)
        adc_lower, adc_upper = get_global_bounds(adc_volume)

        for z in range(depth):
            mask_slice = mask_data[:, :, z]

            # 🔴 严格剔除：如果当前切片没有任何器官标签，直接跳过 (不保留负样本)
            if np.sum(mask_slice) == 0:
                continue

            # ==============================================================
            # === Multi-Modal Fusion (CLIP-Optimized Pseudo-RGB) ===
            # ==============================================================
            t2_slice = img_data[:, :, z, 0]
            adc_slice = img_data[:, :, z, 1]

            # R通道 & G通道: T2 与 ADC 归一化 (包含原本的 CLAHE)
            t2_norm = apply_normalization_with_bounds(t2_slice, t2_lower, t2_upper)
            adc_norm = apply_normalization_with_bounds(adc_slice, adc_lower, adc_upper)

            # B通道: 提取 T2 的高频边缘 (Sobel梯度)，增强 CLIP 模型的边缘感知
            t2_float = t2_norm.astype(np.float32) / 255.0
            grad_x = cv2.Sobel(t2_float, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(t2_float, cv2.CV_32F, 0, 1, ksize=3)
            edge_map = cv2.magnitude(grad_x, grad_y)
            # 缩放回 0-255 uint8 格式
            edge_map_uint8 = np.clip((edge_map / (edge_map.max() + 1e-8)) * 255, 0, 255).astype(np.uint8)

            # 合成三通道 [解剖, 致密度, 边缘]
            img_rgb_slice = np.stack([t2_norm, adc_norm, edge_map_uint8], axis=-1)
            # ==============================================================

            # 物理重采样 + Crop/Pad
            img_processed = resample_and_pad_crop(img_rgb_slice, spacing, TARGET_SPACING, IMG_SIZE, is_mask=False)
            mask_processed = resample_and_pad_crop(mask_slice, spacing, TARGET_SPACING, IMG_SIZE, is_mask=True)

            file_base_id = f"msd_task05_{subject_id}_{z:03d}"

            # Save Image
            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_processed)
            rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            # Process Organs (仅正样本)
            for organ_id, organ_name in TASK05_ORGAN_MAP.items():
                binary_mask = (mask_processed == organ_id).astype(np.uint8)
                pixels_count = np.sum(binary_mask)

                # 🔴 严格剔除：过滤掉无目标的 Mask 或小于设定阈值的噪点 Mask
                if pixels_count < MIN_PIXEL_THRESHOLD:
                    continue

                    # Save Mask
                task_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{task_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                desc_text = TASK05_EXPERT_DESC[organ_name]

                # Standardized Metadata
                meta_list.append({
                    "img_path": rel_img,
                    "mask_path": rel_mask,
                    "modality": "MRI",
                    "description": desc_text,
                    "organ": organ_name,
                    "raw_organ_id": organ_id,
                    "source": "MSD_Task05"
                })

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return []

    return meta_list


# ================= 4. Main =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting MSD Task05 (Prostate) Preprocessing")
    print(f"✨ Strategy: 3D Norm + Spacing Resample (No Negative Samples) | Workers: {NUM_WORKERS}")

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
        print(f"❌ imagesTr not found in {DATASET_ROOT}")
        return

    # Build Tasks
    tasks = []
    img_files = glob.glob(os.path.join(img_folder, "*.nii*"))

    for img_path in img_files:
        fname = os.path.basename(img_path)
        mask_path = os.path.join(lbl_folder, fname)
        if os.path.exists(mask_path):
            sid = fname.split(".")[0]
            if sid.startswith("._"): sid = sid[2:]

            tasks.append((img_path, mask_path, sid, img_dir, mask_dir))

    print(f"✅ Found {len(tasks)} cases")

    all_metadata = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_case, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res: all_metadata.extend(res)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"🎉 Finished! Samples (Positive only): {len(all_metadata)}")
    print(f"📂 Metadata: {json_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()