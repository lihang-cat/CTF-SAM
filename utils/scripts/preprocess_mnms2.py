import os
import glob
import json
import random
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 配置区域 =================

DATASET_ROOT = r"D:\dataset\MnM2\dataset"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MnM2_256_TextGuided_SA_Mini"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

TARGET_SUBJECTS = 40
IMG_SIZE = 256
USE_25D = True
MIN_PIXEL_THRESHOLD = 20

MNMS_ORGAN_MAP = {
    1: "left ventricle",
    2: "myocardium",
    3: "right ventricle"
}

EXPERT_DESC = {
    "left ventricle": "left ventricle cavity, the main pumping chamber of the heart",
    "myocardium": "myocardium, the thick muscle layer of the left ventricle wall",
    "right ventricle": "right ventricle cavity, the chamber that pumps deoxygenated blood to the lungs"
}


# ================= 2. 图像处理核心函数 =================

def normalize_volume_robust(img_volume):
    mask = img_volume > 0
    if np.sum(mask) == 0: return img_volume
    pixels = img_volume[mask]
    lower = np.percentile(pixels, 0.5)
    upper = np.percentile(pixels, 99.5)
    img_volume = np.clip(img_volume, lower, upper)
    if upper == lower: return np.zeros_like(img_volume)
    return (img_volume - lower) / (upper - lower)


def crop_to_foreground(img_3d, gt_3d, margin=16):
    """
    🌟 SOTA 优化：智能非零区域裁剪。
    自动切除四周大面积纯黑背景，使心脏区域在 Resize 时占据更大的有效像素面积。
    """
    # 找到 3D 卷中非零区域的投影
    mask = img_3d > 1e-5
    xy_mask = np.any(mask, axis=2)  # 投影到 XY 平面，保证所有切片的裁剪框一致
    coords = np.argwhere(xy_mask)

    if len(coords) == 0:
        return img_3d, gt_3d

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 预留安全边距
    H, W = xy_mask.shape
    y_min = max(0, y_min - margin)
    y_max = min(H, y_max + margin + 1)
    x_min = max(0, x_min - margin)
    x_max = min(W, x_max + margin + 1)

    # 同步裁剪图像和标签
    return img_3d[y_min:y_max, x_min:x_max, :], gt_3d[y_min:y_max, x_min:x_max, :]


def apply_volume_enhancement(img_3d_norm):
    """🌟 速度优化：全局预计算 CLAHE，避免 2.5D 切片时的重复计算"""
    img_uint8 = (img_3d_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    clahe_vol = np.zeros_like(img_uint8)
    for z in range(img_uint8.shape[2]):
        clahe_vol[:, :, z] = clahe.apply(img_uint8[:, :, z])
    return clahe_vol


def get_25d_slice_fast(clahe_vol, z_index):
    """🌟 极速获取 2.5D 切片，直接取用已增强的体数据"""
    depth = clahe_vol.shape[2]
    idx_prev = max(0, z_index - 1)
    idx_next = min(depth - 1, z_index + 1)
    return np.stack([
        clahe_vol[:, :, idx_prev],
        clahe_vol[:, :, z_index],
        clahe_vol[:, :, idx_next]
    ], axis=-1)


def resize_pad(image, target_size, is_mask=False):
    """🌟 质量优化：原图使用三次样条插值 (CUBIC)，更好保留心肌内膜细节"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant')
    else:
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant')


# ================= 3. 单个样本处理逻辑 =================

def process_mnms2_pair(args):
    img_path, gt_path, img_dir, mask_dir = args
    try:
        nii_img = nib.as_closest_canonical(nib.load(img_path))
        nii_gt = nib.as_closest_canonical(nib.load(gt_path))
        img_3d = nii_img.get_fdata().astype(np.float32)
        gt_3d = nii_gt.get_fdata().astype(np.uint8)

        filename = os.path.basename(img_path)
        parts = filename.split('_')
        subject_id = parts[0]
        phase = "ED" if "ED" in filename else "ES"

        # 🌟 执行 SOTA 裁剪、归一化与全局增强
        img_3d_norm = normalize_volume_robust(img_3d)
        img_3d_crop, gt_3d_crop = crop_to_foreground(img_3d_norm, gt_3d)
        clahe_vol = apply_volume_enhancement(img_3d_crop)

        processed_meta = []
        depth = img_3d_crop.shape[2]

        for z in range(depth):
            mask_slice = gt_3d_crop[:, :, z]
            if np.sum(mask_slice) == 0: continue

            if USE_25D:
                img_final = get_25d_slice_fast(clahe_vol, z)
            else:
                img_final = np.stack([clahe_vol[:, :, z]] * 3, axis=-1)

            img_padded = resize_pad(img_final, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

            # 🌟 逻辑优化：先筛选有效 Mask，确保有货再执行磁盘 I/O，杜绝“孤儿图像”
            valid_masks = []
            unique_vals = np.unique(mask_padded)
            for val in unique_vals:
                if val == 0 or val not in MNMS_ORGAN_MAP: continue
                binary_mask = (mask_padded == val).astype(np.uint8)
                if np.sum(binary_mask) >= MIN_PIXEL_THRESHOLD:
                    valid_masks.append((val, binary_mask))

            # 如果缩放后没有任何器件达标，直接跳过，不保存废图
            if not valid_masks:
                continue

            slice_id = f"mnm2_{subject_id}_{phase}_z{z:03d}"
            img_save_path = os.path.join(img_dir, f"{slice_id}.npy")
            np.save(img_save_path, img_padded)
            rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            for val, binary_mask in valid_masks:
                organ_name = MNMS_ORGAN_MAP[val]
                task_id = f"{slice_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(mask_dir, f"{task_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                processed_meta.append({
                    "img_path": rel_img,
                    "mask_path": rel_mask,
                    "modality": "MRI",
                    "description": EXPERT_DESC[organ_name],
                    "organ": organ_name,
                    "raw_organ_id": int(val),
                    "source": "MnM2"
                })
        return processed_meta
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
        return []


# ================= 4. 主程序 =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)



    all_gts = glob.glob(os.path.join(DATASET_ROOT, "**", "*_gt.nii.gz"), recursive=True)

    subject_map = {}
    for gt_path in all_gts:
        filename = os.path.basename(gt_path)
        if "_SA_" not in filename: continue
        if ("_ED" not in filename) and ("_ES" not in filename): continue

        subject_id = filename.split('_')[0]
        if subject_id not in subject_map:
            subject_map[subject_id] = []
        subject_map[subject_id].append(gt_path)

    all_subjects = list(subject_map.keys())
    print(f"🔍 扫描到 {len(all_subjects)} 个可用病人")

    random.seed(42)
    selected_subjects = random.sample(all_subjects, min(TARGET_SUBJECTS, len(all_subjects)))
    print(f"🎲 已选中病人 ID: {selected_subjects}")

    tasks = []
    for subj_id in selected_subjects:
        gt_paths = subject_map[subj_id]
        for gt_path in gt_paths:
            img_path = gt_path.replace("_gt.nii.gz", ".nii.gz")
            if os.path.exists(img_path):
                tasks.append((img_path, gt_path, img_dir, mask_dir))

    print(f"✅ 生成 {len(tasks)} 个 3D 处理任务")

    final_metadata = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_mnms2_pair, t): t for t in tasks}
        for f in tqdm(as_completed(futures), total=len(tasks)):
            try:
                result = f.result()
                if result:
                    final_metadata.extend(result)
            except Exception as e:
                task_info = futures[f]
                print(f"❌ 进程抛出未捕获异常，图像: {os.path.basename(task_info[0])} -> {e}")

    json_path = os.path.join(OUTPUT_DIR, "test_A_mnms2_mini.json")
    with open(json_path, "w") as f:
        json.dump(final_metadata, f, indent=4)

    print(f"🎉 处理完成！")
    print(f"📄 最终有效任务对: {len(final_metadata)}")
    print(f"💾 列表已保存: {json_path}")


if __name__ == "__main__":
    main()