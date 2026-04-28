import os
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 配置区域 =================
DATASET_ROOT = r"D:\dataset\BraTS2020_TrainingData"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\BraTS_256_MultiModal_RGB_v1"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 50
MIN_TUMOR_AREA = 50
MAX_SLICES_PER_PATIENT = 10
MIN_Z_DISTANCE = 3  # 相邻切片的最小 Z 轴间距

# 🌟 核心优化：彻底改为3模态对应RGB
MODALITIES = ["flair", "t1ce", "t2"]

NUM_WORKERS = 8


# ================= 2. 核心函数 =================
def normalize_slice(img_slice, mean, std):
    """基于全局均值/标准差对单张切片进行快速 Z-Score 归一化"""
    brain_mask = img_slice > 0
    if not np.any(brain_mask) or std < 1e-6:
        return np.zeros_like(img_slice)

    img_norm = np.zeros_like(img_slice)
    img_norm[brain_mask] = (img_slice[brain_mask] - mean) / std

    # 快速切片级百分位数截断，提升对比度
    lower = np.percentile(img_norm[brain_mask], 0.5)
    upper = np.percentile(img_norm[brain_mask], 99.5)

    if upper > lower:
        img_norm = np.clip(img_norm, lower, upper)
        img_norm[brain_mask] = (img_norm[brain_mask] - lower) / (upper - lower)
    else:
        img_norm[brain_mask] = 0

    img_norm[~brain_mask] = 0
    return img_norm


def apply_2d_enhancement(img_slice_norm):
    """自适应CLAHE增强，输出高质量 uint8"""
    brain_mask_slice = img_slice_norm > 0.01
    img_uint8 = (img_slice_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8)
    img_clahe[~brain_mask_slice] = 0
    return img_clahe


def fuse_modalities_to_rgb(flair_slice, t1ce_slice, t2_slice):
    """将三大模态直接映射到RGB通道"""
    return np.stack([flair_slice, t1ce_slice, t2_slice], axis=-1)


def resize_pad(image, target_size, is_mask=False):
    """鲁棒的尺寸调整+填充"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_h, pad_w = target_size - new_h, target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant')
    else:
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant')


def find_file(folder, keyword):
    """极速扁平文件查找"""
    try:
        for f in os.listdir(folder):
            if keyword.lower() in f.lower() and f.endswith(('.nii', '.nii.gz')):
                return os.path.join(folder, f)
    except Exception:
        pass
    return None


# ================= 3. 单病例处理 (延迟加载优化版) =================
def process_single_case(args):
    case_path, output_img_dir, output_mask_dir = args
    subject_id = os.path.basename(case_path)

    # 1. 优先只读 SEG 文件，计算 NMS 选取最优切片
    path_seg = find_file(case_path, "seg") or find_file(case_path, "ot")
    if not path_seg:
        return []
    seg_vol = nib.load(path_seg).get_fdata().astype(np.uint8)

    tumor_mask = (seg_vol > 0)
    enhancing_mask = (seg_vol == 4)

    areas = np.sum(tumor_mask, axis=(0, 1))
    enhancing_areas = np.sum(enhancing_mask, axis=(0, 1))

    weights = 1.0 + enhancing_areas / (areas + 1e-6)
    scores = areas * weights
    valid_indices = np.where(areas > MIN_TUMOR_AREA)[0]

    if len(valid_indices) == 0:
        return []

    valid_scores = scores[valid_indices]
    sorted_order = np.argsort(valid_scores)[::-1]
    sorted_valid_indices = valid_indices[sorted_order]

    selected_indices = []

    # NMS 逻辑
    for idx in sorted_valid_indices:
        if len(selected_indices) >= MAX_SLICES_PER_PATIENT: break
        if all(abs(idx - selected) >= MIN_Z_DISTANCE for selected in selected_indices):
            selected_indices.append(idx)

    # 兜底补齐
    if len(selected_indices) < min(MAX_SLICES_PER_PATIENT, len(valid_indices)):
        for idx in sorted_valid_indices:
            if len(selected_indices) >= min(MAX_SLICES_PER_PATIENT, len(valid_indices)): break
            if idx not in selected_indices:
                selected_indices.append(idx)

    target_indices = sorted(selected_indices)

    if not target_indices:
        return []

    # 2. 只有确定了要提取的切片后，才去计算模态数据的全局均值和方差
    mod_data = {}
    for mod in MODALITIES:
        path = find_file(case_path, mod)
        if not path: return []
        vol = nib.load(path).get_fdata().astype(np.float32)

        # 极速全局脑区掩码 (替代之前耗时的 create_brain_mask)
        brain_mask_global = vol > 0
        brain_pixels = vol[brain_mask_global]

        mean = np.mean(brain_pixels) if len(brain_pixels) > 0 else 0
        std = np.std(brain_pixels) if len(brain_pixels) > 0 else 1

        mod_data[mod] = {"vol": vol, "mean": mean, "std": std}

    case_meta_list = []

    # 3. 仅对选中的优选切片进行处理
    for z in target_indices:
        # 读取原始掩码切片并 Resize
        mask_slice_raw = seg_vol[:, :, z]
        mask_padded = resize_pad(mask_slice_raw, IMG_SIZE, is_mask=True)

        # 处理图像切片
        flair_norm = normalize_slice(mod_data["flair"]["vol"][:, :, z], mod_data["flair"]["mean"],
                                     mod_data["flair"]["std"])
        t1ce_norm = normalize_slice(mod_data["t1ce"]["vol"][:, :, z], mod_data["t1ce"]["mean"], mod_data["t1ce"]["std"])
        t2_norm = normalize_slice(mod_data["t2"]["vol"][:, :, z], mod_data["t2"]["mean"], mod_data["t2"]["std"])

        flair_slice = apply_2d_enhancement(flair_norm)
        t1ce_slice = apply_2d_enhancement(t1ce_norm)
        t2_slice = apply_2d_enhancement(t2_norm)

        img_clip_rgb_raw = fuse_modalities_to_rgb(flair_slice, t1ce_slice, t2_slice)
        img_clip_rgb = resize_pad(img_clip_rgb_raw, IMG_SIZE, is_mask=False)

        # 保存伪 RGB 图像
        file_base = f"brats_{subject_id}_{z:03d}"
        img_save_path = os.path.join(output_img_dir, f"{file_base}.npy")
        np.save(img_save_path, img_clip_rgb)
        rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

        # ================= 🌟 临床复合标签处理逻辑 =================
        composite_regions = {
            "Whole_Tumor": {
                "mask_data": ((mask_padded == 1) | (mask_padded == 2) | (mask_padded == 4)).astype(np.uint8),
                "desc": "The whole brain tumor region, including the necrotic core, peritumoral edema, and enhancing tumor."
            },
            "Tumor_Core": {
                "mask_data": ((mask_padded == 1) | (mask_padded == 4)).astype(np.uint8),
                "desc": "The tumor core region, representing the bulk of the tumor mass excluding the peritumoral edema."
            },
            "Enhancing_Tumor": {
                "mask_data": (mask_padded == 4).astype(np.uint8),
                "desc": "The actively enhancing brain tumor region, indicating high vascularity and active tumor tissue."
            }
        }

        for region_name, region_info in composite_regions.items():
            binary_mask = region_info["mask_data"]

            # 过滤无效噪点掩码
            if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD:
                continue

            # 保存独立的二值化复合掩码
            mask_file_base = f"{file_base}_{region_name}"
            mask_save_path = os.path.join(output_mask_dir, f"{mask_file_base}.npy")
            np.save(mask_save_path, binary_mask)
            rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

            # 对齐 Metadata
            case_meta_list.append({
                "img_path": rel_img_path,
                "mask_path": rel_mask_path,
                "modality": "MRI_PseudoRGB",
                "description": region_info["desc"],
                "organ": region_name.replace("_", " "),
                "raw_organ_id": -1,
                "source": "BraTS"
            })
        # ===============================================================

    return case_meta_list


# ================= 4. 主程序 =================
def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    patient_dirs = set()
    for root, dirs, files in os.walk(DATASET_ROOT):
        lower_files = [f.lower() for f in files]
        has_all_modalities = all(any(mod in f for f in lower_files) for mod in MODALITIES)
        has_seg = any('seg' in f for f in lower_files) or any('ot' in f for f in lower_files)
        if has_all_modalities and has_seg:
            patient_dirs.add(root)

    patient_dirs = list(patient_dirs)
    print(f"🚀 发现 {len(patient_dirs)} 个有效病例（含所需模态+分割标签）")

    all_meta_data = []
    tasks = [(path, img_dir, mask_dir) for path in patient_dirs]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_case, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理病例"):
            case_path = futures[future][0]
            try:
                result = future.result()
                if result:
                    all_meta_data.extend(result)
            except Exception as e:
                print(f"❌ 病例 {os.path.basename(case_path)} 处理失败：{str(e)}")
                continue

    # 去重逻辑确保稳定
    unique_meta = []
    seen_keys = set()
    for meta in all_meta_data:
        key = (meta["img_path"], meta["mask_path"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_meta.append(meta)

    meta_save_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(unique_meta, f, indent=4, ensure_ascii=False)

    print(f"🎉 预处理完成！")
    print(f"📊 生成 {len(unique_meta)} 个CLIP适配样本及二值化复合掩码标签")
    print(f"📂 输出目录：{OUTPUT_DIR}")
    print(f"📄 元数据文件：{meta_save_path}")


if __name__ == "__main__":
    main()