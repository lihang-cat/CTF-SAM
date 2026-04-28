import os
import glob
import random
import json
import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 配置区域 =================

DATASET_ROOT = r"D:\dataset\CHAOS_Train_Sets\Train_Sets\MR"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\CHAOS_MRI_256_Expert_2.5D_Simple"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# 参数
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20
NUM_WORKERS = 8
USE_25D = True  # 2.5D 依然保留，这对分割精度很有帮助

# ID 列表
MR_IDS = [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]

# 灰度映射
MR_ORGAN_RANGES = {
    "liver": (55, 70),
    "right kidney": (110, 135),
    "left kidney": (175, 200),
    "spleen": (240, 255)
}

# 简化后的专家描述 (去掉了序列特定的描述)
MR_EXPERT_DESC = {
    "liver": "liver, a large organ in the upper right abdomen",
    "right kidney": "right kidney, situated in the retroperitoneal space",
    "left kidney": "left kidney, situated in the retroperitoneal space",
    "spleen": "spleen, located in the upper left abdomen"
}


# ================= 2. 核心函数 =================

def read_dicom_series_robust(folder_path):
    """ 鲁棒读取 DICOM """
    if not os.path.exists(folder_path): return None
    reader = sitk.ImageSeriesReader()
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
        if not dicom_names:
            dicom_names = sorted(glob.glob(os.path.join(folder_path, "*")))
            dicom_names = [f for f in dicom_names if os.path.isfile(f) and not f.endswith('.png')]
        if not dicom_names: return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return sitk.GetArrayFromImage(image)
    except Exception:
        return None


def read_png_masks(ground_folder):
    """ 读取 Mask """
    if not os.path.exists(ground_folder): return None
    png_files = sorted(glob.glob(os.path.join(ground_folder, "*.png")))
    if not png_files: return None
    masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in png_files]
    return np.array(masks)


def normalize_mri_zscore(img_slice):
    """ Z-Score 归一化 """
    mask = img_slice > 0
    if np.sum(mask) == 0: return img_slice.astype(np.uint8)
    pixels = img_slice[mask]
    mean = np.mean(pixels)
    std = np.std(pixels)
    if std == 0: return np.zeros_like(img_slice, dtype=np.uint8)
    img_norm = (img_slice - mean) / std
    img_norm = np.clip(img_norm, -3, 3)
    img_norm = (img_norm + 3) / 6.0
    img_norm[img_slice == 0] = 0
    return img_norm


def apply_2d_enhancement(img_slice_norm):
    img_uint8 = (img_slice_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def get_25d_slice(vol_data, z_index):
    """ 2.5D 堆叠 """
    depth = vol_data.shape[0]
    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    img_prev = apply_2d_enhancement(normalize_mri_zscore(vol_data[idx_prev]))
    img_curr = apply_2d_enhancement(normalize_mri_zscore(vol_data[idx_curr]))
    img_next = apply_2d_enhancement(normalize_mri_zscore(vol_data[idx_next]))

    return np.stack([img_prev, img_curr, img_next], axis=-1)


def resize_pad(image, target_size, is_mask=False):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    pad_val = 0
    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant', constant_values=pad_val)
    else:
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=pad_val)


def find_dicom_dirs(base_dicom_path):
    """ 寻找 DICOM 路径 """
    target_dirs = []
    sub_dirs = [d for d in os.listdir(base_dicom_path) if os.path.isdir(os.path.join(base_dicom_path, d))]
    valid_subs = [d for d in sub_dirs if "Phase" in d]
    if valid_subs:
        for sub in valid_subs: target_dirs.append((sub, os.path.join(base_dicom_path, sub)))
    else:
        target_dirs.append(("Standard", base_dicom_path))
    return target_dirs


# ================= 3. 处理逻辑 =================

def process_single_patient(args):
    pid, mr_type, output_img_dir, output_mask_dir = args
    meta_list = []

    subject_path = os.path.join(DATASET_ROOT, str(pid), mr_type)
    dicom_root = os.path.join(subject_path, "DICOM_anon")
    ground_dir = os.path.join(subject_path, "Ground")

    if not os.path.exists(dicom_root): return []

    mask_vol = read_png_masks(ground_dir)
    if mask_vol is None: return []

    target_dirs = find_dicom_dirs(dicom_root)

    for phase_name, real_dicom_path in target_dirs:
        vol_data = read_dicom_series_robust(real_dicom_path)
        if vol_data is None: continue
        if vol_data.shape[0] != mask_vol.shape[0]: continue

        unique_id = f"{pid}_{mr_type}_{phase_name}"
        z_slices = vol_data.shape[0]

        # 🌟🌟 简化点：统一使用 "MRI" 作为模态，不再区分序列 🌟🌟
        modality_str = "MRI"

        for z in range(z_slices):
            mask_slice = mask_vol[z, :, :]
            if np.sum(mask_slice) == 0: continue

            if USE_25D:
                img_processed = get_25d_slice(vol_data, z)
            else:
                slice_enhanced = apply_2d_enhancement(normalize_mri_zscore(vol_data[z]))
                img_processed = np.stack([slice_enhanced] * 3, axis=-1)

            img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

            file_base_id = f"chaos_mri_{unique_id}_{z:03d}"

            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_padded)
            rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            has_valid_organ = False
            for organ, (low, high) in MR_ORGAN_RANGES.items():
                binary_mask = ((mask_padded >= low) & (mask_padded <= high)).astype(np.uint8)
                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

                task_file_id = f"{file_base_id}_{organ.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{task_file_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                desc_text = MR_EXPERT_DESC[organ]

                # ✅ 标准化、简化的元数据
                meta_list.append({
                    "img_path": rel_img_path,
                    "mask_path": rel_mask_path,
                    "modality": modality_str,  # 🌟 统一为 "MRI"
                    "description": desc_text,
                    "organ": organ,
                    "raw_organ_id": int(low),
                    "source": "CHAOS_MRI"
                })
                has_valid_organ = True

            if not has_valid_organ and os.path.exists(img_save_path):
                os.remove(img_save_path)

    return meta_list


def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 开始 CHAOS MRI 预处理 (简化版)")

    tasks = []
    mr_types = ["T1DUAL", "T2SPIR"]
    for pid in MR_IDS:
        for m_type in mr_types:
            tasks.append((pid, m_type, img_dir, mask_dir))

    all_meta_data = []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_patient, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res: all_meta_data.extend(res)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=4)

    print(f"🎉 完成! 元数据: {json_path}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 路径不存在: {DATASET_ROOT}")
    else:
        main()