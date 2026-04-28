import os
import random
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 配置区域 =================

DATASET_ROOT = r"D:\dataset\MSD\Task02_Heart"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MSD_Task02_256_2.5D_Expert"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
USE_25D = True  # 🌟 开启 2.5D 堆叠
NUM_WORKERS = 8  # ⚡ 进程数 (建议设为 CPU 核心数的一半或 4-8)

# ================= 2. Task02 专属定义 =================

TASK02_ORGAN_MAP = {1: "left atrium"}

TASK02_EXPERT_DESC = {
    1: "left atrium, the upper chamber of the heart that receives oxygenated blood from the lungs"
}


# ================= 3. 核心函数 =================

def normalize_volume_3d(img_volume):
    """ 3D 级一致性归一化 """
    mask = img_volume > 0
    if np.sum(mask) == 0: return img_volume

    pixels = img_volume[mask]
    lower = np.percentile(pixels, 0.5)
    upper = np.percentile(pixels, 99.5)

    img_volume = np.clip(img_volume, lower, upper)
    if upper == lower: return np.zeros_like(img_volume)

    img_volume = (img_volume - lower) / (upper - lower)
    return img_volume


def apply_2d_enhancement(img_slice_norm):
    """ CLAHE 增强 """
    img_uint8 = (img_slice_norm * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_uint8)


def get_25d_slice(volume, z_index):
    """ 🌟 获取 2.5D 切片 [z-1, z, z+1] """
    depth = volume.shape[2]
    slice_curr = volume[:, :, z_index]

    idx_prev = max(0, z_index - 1)
    slice_prev = volume[:, :, idx_prev]

    idx_next = min(depth - 1, z_index + 1)
    slice_next = volume[:, :, idx_next]

    img_prev = apply_2d_enhancement(slice_prev)
    img_curr = apply_2d_enhancement(slice_curr)
    img_next = apply_2d_enhancement(slice_next)

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

    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant')
    else:
        if not is_mask:
            resized = np.clip(resized, 0, 255).astype(np.uint8)
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant')


def process_single_case(args):
    """
    单个病例处理函数 (必须是顶层函数以支持多进程 pickle)
    """
    img_path, mask_path, subject_id, output_img_dir, output_mask_dir = args
    case_meta_list = []

    try:
        # 1. 加载并统一方向
        img_nii = nib.as_closest_canonical(nib.load(img_path))
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))

        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.uint8)

        # 维度修正
        if img_data.ndim == 4: img_data = img_data[:, :, :, 0]
        if mask_data.ndim == 4: mask_data = mask_data[:, :, :, 0]

        # 2. 3D 归一化
        img_data_norm = normalize_volume_3d(img_data)
        depth = img_data.shape[2]

        for z in range(depth):
            mask_slice = mask_data[:, :, z]

            # 过滤空切片
            if np.sum(mask_slice) == 0: continue

            # === 图像处理 (2.5D) ===
            if USE_25D:
                img_final = get_25d_slice(img_data_norm, z)
            else:
                img_slice_processed = apply_2d_enhancement(img_data_norm[:, :, z])
                img_final = np.stack([img_slice_processed] * 3, axis=-1)

            # Resize + Pad
            img_padded = resize_pad(img_final, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

            file_base_id = f"msd_task02_{subject_id}_{z:03d}"

            # 保存图像
            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_padded)
            rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            # === Label 处理 ===
            unique_vals = np.unique(mask_padded)
            for val in unique_vals:
                if val != 1: continue

                binary_mask = (mask_padded == val).astype(np.uint8)
                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

                organ_name = TASK02_ORGAN_MAP[val]

                # 获取详细描述
                desc_text = TASK02_EXPERT_DESC.get(int(val), organ_name)

                # 保存 Mask
                task_file_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{task_file_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                # ✅ 按照新 Dataset 格式修改 Append
                case_meta_list.append({
                    "img_path": rel_img_path,
                    "mask_path": rel_mask_path,
                    "modality": "MRI",  # 🌟 属性 1: 模态
                    "description": desc_text,  # 🌟 属性 2: 详细描述
                    "organ": organ_name,  # 🌟 属性 3: 器官名
                    "raw_organ_id": int(val),
                    "source": "MSD_Task02"
                })

    except Exception as e:
        print(f"❌ Error {subject_id}: {e}")
        return []

    return case_meta_list


def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # 路径查找
    img_folder = os.path.join(DATASET_ROOT, "imagesTr")
    lbl_folder = os.path.join(DATASET_ROOT, "labelsTr")

    if not os.path.exists(img_folder):
        for root, dirs, files in os.walk(DATASET_ROOT):
            if "imagesTr" in dirs:
                img_folder = os.path.join(root, "imagesTr")
                lbl_folder = os.path.join(root, "labelsTr")
                break

    print(f"🚀 开始多进程预处理 | Workers: {NUM_WORKERS} | 2.5D: {USE_25D}")

    img_files = [f for f in os.listdir(img_folder) if f.endswith(".nii") or f.endswith(".nii.gz")]
    img_files = [f for f in img_files if not f.startswith("._")]
    img_files.sort()

    # 准备任务列表
    tasks = []
    for filename in img_files:
        img_path = os.path.join(img_folder, filename)
        mask_path = os.path.join(lbl_folder, filename)
        if not os.path.exists(mask_path): continue

        subject_id = filename.split(".")[0]
        tasks.append((img_path, mask_path, subject_id, img_dir, mask_dir))

    all_meta_data = []

    # ⚡ 启动多进程池
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 提交任务
        futures = [executor.submit(process_single_case, task) for task in tasks]

        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            if result:
                all_meta_data.extend(result)

    # 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=4)

    print(f"🎉 完成! 共生成 {len(all_meta_data)} 个样本.")
    print(f"📂 数据位置: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Windows 下必须放在 if __name__ == "__main__": 之下
    main()