import os
import glob
import json
import numpy as np
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================
DATASET_ROOT = r"D:\dataset\CHAOS_Train_Sets\Train_Sets\CT"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\CHAOS_CT_256_Expert"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# 🌟 Strategy Toggle
USE_SPATIAL_25D = False
USE_MASK_CLEANUP = True  # 开启掩码净化

# CHAOS CT Official IDs
CT_IDS = [1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# ================= 2. Definitions =================
CHAOS_ORGAN_MAP = {1: "liver"}
CHAOS_EXPERT_DESC = {
    "liver": "liver, a large wedge-shaped organ in the upper right quadrant of the abdomen, appearing homogeneous on CT"
}

# 🌟 核心策略：多窗宽窗位配置（肝脏、软组织、骨骼）
WINDOW_CONFIGS = {
    "liver": (30, 150),
    "soft": (40, 400),
    "bone": (100, 1000)
}


# ================= 3. Core Functions =================
def read_dicom_series(folder_path):
    """ 🌟 修复：强行按文件名排序，确保与 Ground 文件夹中的 PNG 掩码绝对对齐 """
    if not os.path.exists(folder_path):
        return None
    reader = sitk.ImageSeriesReader()
    try:
        # 舍弃 GDCM 的物理排序，直接用 glob 的文件名字母排序保证与 mask 一致
        dicom_names = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
        if not dicom_names:
            return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        # 保持原始 HU 值
        return sitk.GetArrayFromImage(image).astype(np.float32)
    except Exception as e:
        print(f"❌ 读取DICOM失败 {folder_path}: {e}")
        return None


def read_png_masks(ground_folder):
    """ 使用 cv2 加速读取掩码 """
    if not os.path.exists(ground_folder): return None
    png_files = sorted(glob.glob(os.path.join(ground_folder, "*.png")))
    if not png_files: return None

    masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in png_files]
    valid_masks = [m for m in masks if m is not None]

    return np.array(valid_masks) if valid_masks else None


def apply_window(img, center, width):
    """ 🌟 修复：移除全局截断，让每个窗宽窗位自行管辖数值范围 """
    lower = center - width / 2.0
    upper = center + width / 2.0
    img_clamped = np.clip(img, lower, upper)

    if upper == lower:
        return np.zeros_like(img, dtype=np.uint8)

    windowed = (img_clamped - lower) / (upper - lower)
    # 映射到 0-255 并转换为 uint8
    return (windowed * 255).astype(np.uint8)


def get_multi_window_slice(img_slice):
    """ 🌟 核心：生成 3 通道伪彩 CT (R: 肝脏窗, G: 软组织窗, B: 骨窗) """
    ch_r = apply_window(img_slice, *WINDOW_CONFIGS["liver"])
    ch_g = apply_window(img_slice, *WINDOW_CONFIGS["soft"])
    ch_b = apply_window(img_slice, *WINDOW_CONFIGS["bone"])
    return np.stack([ch_r, ch_g, ch_b], axis=-1)


def get_spatial_25d_slice(vol_data, z_index):
    depth = vol_data.shape[0]
    w_center, w_width = WINDOW_CONFIGS["liver"]
    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    slice_prev = apply_window(vol_data[idx_prev], w_center, w_width)
    slice_curr = apply_window(vol_data[idx_curr], w_center, w_width)
    slice_next = apply_window(vol_data[idx_next], w_center, w_width)
    return np.stack([slice_prev, slice_curr, slice_next], axis=-1)


def resize_pad(image, target_size, is_mask=False):
    """ 使用 cv2.copyMakeBorder 替换 np.pad，提速处理 """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    delta_h, delta_w = target_size - new_h, target_size - new_w
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padding_value = 0 if is_mask else [0, 0, 0]
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)


def clean_mask(mask):
    """ 使用 OpenCV 重写，速度提升 3-5 倍 """
    binary = (mask > 127).astype(np.uint8)
    if cv2.countNonZero(binary) < MIN_PIXEL_THRESHOLD:
        return np.zeros_like(binary)

    # 1. 形态学闭运算 (先填补小孔洞)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 2. 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:  # 只有背景
        return np.zeros_like(binary)

    # 3. 提取面积最大的连通域 (排除背景 label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest = (labels == largest_label).astype(np.uint8)

    return largest if cv2.countNonZero(largest) >= MIN_PIXEL_THRESHOLD else np.zeros_like(largest)


def process_single_patient(args):
    pid, output_img_dir, output_mask_dir = args
    meta_list = []

    subject_path = os.path.join(DATASET_ROOT, str(pid))
    dicom_dir = os.path.join(subject_path, "DICOM_anon")
    ground_dir = os.path.join(subject_path, "Ground")

    vol_data = read_dicom_series(dicom_dir)
    mask_vol = read_png_masks(ground_dir)

    if vol_data is None or mask_vol is None:
        return []

    min_slices = min(vol_data.shape[0], mask_vol.shape[0])

    for z in range(min_slices):
        mask_slice = mask_vol[z]

        # 掩码净化
        if USE_MASK_CLEANUP:
            mask_clean = clean_mask(mask_slice)
            if cv2.countNonZero(mask_clean) == 0:
                continue
        else:
            mask_clean = (mask_slice > 127).astype(np.uint8)
            if cv2.countNonZero(mask_clean) < MIN_PIXEL_THRESHOLD:
                continue

        # 图像处理
        if USE_SPATIAL_25D:
            img_processed = get_spatial_25d_slice(vol_data, z)
        else:
            img_processed = get_multi_window_slice(vol_data[z])

        # Resize & Pad
        img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask_clean, IMG_SIZE, is_mask=True)

        file_base_id = f"chaos_ct_{pid}_{z:03d}"
        organ_name = "liver"
        task_file_id = f"{file_base_id}_{organ_name}"

        img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
        mask_save_path = os.path.join(output_mask_dir, f"{task_file_id}.npy")

        # 统一保存为 uint8，大幅节省磁盘和读写耗时
        np.save(img_save_path, img_padded)
        np.save(mask_save_path, mask_padded)

        meta_list.append({
            "img_path": os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/"),
            "mask_path": os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/"),
            "modality": "CT",
            "description": CHAOS_EXPERT_DESC[organ_name],
            "organ": organ_name,
            "raw_organ_id": 1,
            "source": "CHAOS_CT"
        })

    return meta_list


def main():
    # 增加 Windows 多进程冻结支持
    multiprocessing.freeze_support()

    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting CHAOS CT Preprocessing")
    print(f"✨ Strategy: {'Spatial 2.5D' if USE_SPATIAL_25D else 'Multi-Window'} | Cleanup: {USE_MASK_CLEANUP}")
    print(f"⚡ Workers: {NUM_WORKERS}")

    tasks = [(pid, img_dir, mask_dir) for pid in CT_IDS]
    all_meta_data = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 🌟 修复：建立任务映射，加上 try-except 保护，防止单病例报错崩掉全盘
        futures = {executor.submit(process_single_patient, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            try:
                res = future.result()
                if res:
                    all_meta_data.extend(res)
            except Exception as e:
                pid = futures[future][0]
                print(f"❌ 病例 {pid} 处理崩溃: {str(e)}")

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=4)

    print(f"🎉 Finished! Total samples: {len(all_meta_data)}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Dataset root not found: {DATASET_ROOT}")
    else:
        main()