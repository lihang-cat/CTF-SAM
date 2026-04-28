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
RAW_DATA_ROOT = r"D:\dataset\CAMUS_public\database_nifti"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\CAMUS_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# 🔴 参数
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20
NUM_WORKERS = 8  # 多进程数量

# ================= 2. 定义区域 =================

LABEL_MAP = {
    1: "left ventricle",
    2: "myocardium",
    3: "left atrium"
}

# 专家级描述
CAMUS_EXPERT_DESC = {
    1: "left ventricle cavity, appearing anechoic (dark) and enclosed by the myocardium",
    2: "myocardium, the hyperechoic (bright) muscle wall of the left ventricle",
    3: "left atrium cavity, the anechoic (dark) chamber above the ventricle"
}


# ================= 3. 核心函数 =================

def preprocess_ultrasound(img_data):
    """
    🌟 超声增强流水线:
    NaN处理 -> 鲁棒截断 -> 归一化 -> 双边滤波去噪 -> CLAHE
    """
    img_data = np.nan_to_num(img_data).astype(float)

    # 鲁棒截断 (1% - 99%)
    lower = np.percentile(img_data, 1.0)
    upper = np.percentile(img_data, 99.0)
    img_data = np.clip(img_data, lower, upper)

    # 归一化到 0-255
    if upper == lower:
        img_uint8 = np.zeros_like(img_data, dtype=np.uint8)
    else:
        img_data = (img_data - lower) / (upper - lower)
        img_uint8 = (img_data * 255).astype(np.uint8)

    # 双边滤波去噪 (保留边缘)
    denoised = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=75, sigmaSpace=75)

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def resize_pad(image, target_size, is_mask=False):
    """ 保持比例缩放 + 居中补零 """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    padding_config = ((top, bottom), (left, right))
    if not is_mask and image.ndim == 3:  # RGB
        padding_config = ((top, bottom), (left, right), (0, 0))

    return np.pad(resized, padding_config, mode='constant', constant_values=0)


def process_single_patient(args):
    """
    多进程 Worker
    """
    patient_dir, output_img_dir, output_mask_dir = args
    patient_id = os.path.basename(patient_dir)
    meta_list = []

    # 查找所有 GT 文件
    gt_files = glob.glob(os.path.join(patient_dir, "*_gt.nii.gz"))

    for gt_path in gt_files:
        try:
            filename = os.path.basename(gt_path)

            if "sequence" in filename: continue

            # 解析文件名: patient0001_2CH_ED_gt.nii.gz
            parts = filename.replace(".nii.gz", "").split("_")
            if len(parts) < 4: continue

            view = parts[1]  # 2CH / 4CH
            phase = parts[2]  # ED / ES

            img_filename = filename.replace("_gt.nii.gz", ".nii.gz")
            img_path = os.path.join(patient_dir, img_filename)

            if not os.path.exists(img_path): continue

            # 加载 NIfTI
            mask_obj = nib.load(gt_path)
            img_obj = nib.load(img_path)

            mask_data = np.squeeze(mask_obj.get_fdata())
            img_data = np.squeeze(img_obj.get_fdata())

            if img_data.ndim == 3: img_data = img_data[:, :, 0]
            if mask_data.ndim == 3: mask_data = mask_data[:, :, 0]

            # 旋转校正
            img_data = np.rot90(img_data, k=1)
            mask_data = np.rot90(mask_data, k=1)

            # 预处理
            img_processed = preprocess_ultrasound(img_data)
            img_rgb = np.stack([img_processed] * 3, axis=-1)

            # Resize & Pad
            img_padded = resize_pad(img_rgb, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(mask_data, IMG_SIZE, is_mask=True)

            file_base_id = f"camus_{patient_id}_{view}_{phase}"

            # 保存图像
            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_padded)
            rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            # 🌟🌟 核心修改：构建详细的模态描述字符串 🌟🌟
            # 将 2CH/4CH 和 ED/ES 扩展为自然语言
            view_str = "2-chamber" if view == "2CH" else "4-chamber"
            phase_str = "End-Diastole" if phase == "ED" else "End-Systole"

            # 组合成新的模态字段: e.g., "2-chamber End-Diastole Echocardiogram"
            modality_text = f"{view_str} {phase_str} Echocardiogram"

            # 处理每个器官
            unique_vals = np.unique(mask_padded)
            for val in unique_vals:
                val = int(val)
                if val == 0 or val not in LABEL_MAP: continue

                organ_name = LABEL_MAP[val]
                binary_mask = (mask_padded == val).astype(np.uint8)

                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

                # 保存 Mask
                mask_save_name = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{mask_save_name}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                desc_text = CAMUS_EXPERT_DESC.get(val, organ_name)

                # ✅ 标准元数据格式
                meta_list.append({
                    "img_path": rel_img_path,
                    "mask_path": rel_mask_path,
                    "modality": modality_text,  # 🌟 修改点：融合了视口和时相信息的模态
                    "description": desc_text,
                    "organ": organ_name,
                    "raw_organ_id": val,
                    "source": "CAMUS"
                })

        except Exception as e:
            # print(f"❌ Error {filename}: {e}")
            pass

    return meta_list


def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if not os.path.exists(RAW_DATA_ROOT):
        print(f"❌ 路径不存在: {RAW_DATA_ROOT}")
        return

    patient_dirs = sorted(glob.glob(os.path.join(RAW_DATA_ROOT, "patient*")))
    print(f"🚀 开始 CAMUS 多进程预处理")
    print(f"✨ 策略: 融合视口信息到模态 + 双边滤波去噪")
    print(f"✅ 找到 {len(patient_dirs)} 个病人")

    all_meta_data = []
    tasks = [(p_dir, img_dir, mask_dir) for p_dir in patient_dirs]

    # ⚡ 启动进程池
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_patient, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            if result:
                all_meta_data.extend(result)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=4)

    print(f"🎉 处理完成! 共生成 {len(all_meta_data)} 个样本")
    print(f"📂 元数据保存至: {json_path}")


if __name__ == "__main__":
    main()