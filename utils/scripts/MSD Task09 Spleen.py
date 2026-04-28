import os
import glob
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. 配置区域 =================

DATASET_ROOT = r"D:\dataset\MSD\Task09_Spleen"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MSD_Task09_Spleen_256_2.5D"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 50  # 严格执行 50 像素阈值
USE_25D = True  # 开启 2.5D
NUM_WORKERS = 8

# ================= 2. 映射与描述 =================

# MSD Task09 标签: 0=背景, 1=脾脏
ORGAN_NAME = "spleen"


def get_description():
    # 专家级脾脏解剖描述，提供给语言模型
    return "spleen, a highly vascular, solid organ located in the upper left quadrant of the abdomen, appearing homogeneous on CT"


# ================= 3. 核心处理函数 =================

def apply_window(img, center, width):
    """ CT 窗宽窗位截断并归一化到 [0, 1] """
    lower = center - width / 2.0
    upper = center + width / 2.0
    img_clamped = np.clip(img, lower, upper)
    if upper == lower: return np.zeros_like(img)
    return (img_clamped - lower) / (upper - lower)


def get_25d_slice(vol_data, z_index):
    """ 2.5D 策略: 使用标准腹部软组织窗 (WL:40, WW:400) """
    depth = vol_data.shape[2]
    W_CENTER, W_WIDTH = 40, 400  # 适合脾脏/肝脏等软组织的经典窗宽窗位

    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    s_prev = apply_window(vol_data[:, :, idx_prev], W_CENTER, W_WIDTH)
    s_curr = apply_window(vol_data[:, :, idx_curr], W_CENTER, W_WIDTH)
    s_next = apply_window(vol_data[:, :, idx_next], W_CENTER, W_WIDTH)

    # 堆叠成 (H, W, 3) 并转为 uint8
    return (np.stack([s_prev, s_curr, s_next], axis=-1) * 255).astype(np.uint8)


def resize_pad(image, target_size, is_mask=False):
    """ 保持长宽比缩放并 Padding """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    val = 0 if is_mask else [0, 0, 0]
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=val)


def process_single_subject(args):
    img_path, mask_path, subject_id, output_img_dir, output_mask_dir = args
    meta_list = []

    try:
        # 加载 NIfTI 图像并修正方向为 RAS
        ct_nii = nib.as_closest_canonical(nib.load(img_path))
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))

        # 使用 float32 节省内存
        ct_vol = np.asanyarray(ct_nii.dataobj, dtype=np.float32)
        mask_vol = np.asanyarray(mask_nii.dataobj, dtype=np.uint8)

        if mask_vol.shape != ct_vol.shape:
            print(f"⚠️ 形状不匹配跳过 {subject_id}: CT {ct_vol.shape} vs Mask {mask_vol.shape}")
            return []

        # 快速定位包含脾脏的切片
        z_sums = np.sum(mask_vol, axis=(0, 1))
        valid_slices = np.where(z_sums > 0)[0]

        for z in valid_slices:
            mask_slice = mask_vol[:, :, z]

            # 提取脾脏 (Label 1)
            binary_mask = (mask_slice == 1).astype(np.uint8)

            # ⭐ 严格阈值过滤
            if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD:
                continue

            # 提取图像
            if USE_25D:
                img_processed = get_25d_slice(ct_vol, z)
                modality_str = "CT "
            else:
                norm = apply_window(ct_vol[:, :, z], 40, 400)
                img_processed = np.stack([(norm * 255).astype(np.uint8)] * 3, axis=-1)
                modality_str = "CT"

            # Resize & Pad
            img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(binary_mask, IMG_SIZE, is_mask=True)

            # 生成唯一文件名
            file_base_id = f"msd_spleen_{subject_id}_{z:03d}"

            img_path_save = os.path.join(output_img_dir, f"{file_base_id}.npy")
            mask_path_save = os.path.join(output_mask_dir, f"{file_base_id}.npy")

            # 存储数组
            np.save(img_path_save, img_padded)
            np.save(mask_path_save, mask_padded)

            # 生成 Metadata
            meta_list.append({
                "img_path": os.path.relpath(img_path_save, PROJECT_ROOT).replace("\\", "/"),
                "mask_path": os.path.relpath(mask_path_save, PROJECT_ROOT).replace("\\", "/"),
                "modality": modality_str,
                "description": get_description(),
                "organ": ORGAN_NAME,
                "raw_organ_id": 1,
                "source": "MSD_Task09"
            })

    except Exception as e:
        print(f"❌ Error {subject_id}: {e}")
        return []

    return meta_list


# ================= 4. Main =================

def main():
    # 冻结支持，Windows 多进程必备
    multiprocessing.freeze_support()

    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting MSD Task09 (Spleen) Preprocessing")
    print(f"📉 Strategy: 2.5D Fusion + Strict Threshold > {MIN_PIXEL_THRESHOLD}px")

    # 智能定位图像和标签文件夹
    images_folder = os.path.join(DATASET_ROOT, "imagesTr")
    labels_folder = os.path.join(DATASET_ROOT, "labelsTr")

    if not os.path.exists(images_folder):
        print("🔍 正在递归搜索 imagesTr...")
        found = glob.glob(os.path.join(DATASET_ROOT, "**", "imagesTr"), recursive=True)
        if found:
            images_folder = found[0]
            labels_folder = images_folder.replace("imagesTr", "labelsTr")
        else:
            print(f"❌ 致命错误: 未找到 imagesTr 目录！")
            return

    # 收集任务
    img_files = [f for f in glob.glob(os.path.join(images_folder, "*.nii*")) if
                 not os.path.basename(f).startswith("._")]
    tasks = []

    for img_path in img_files:
        fname = os.path.basename(img_path)
        mask_path = os.path.join(labels_folder, fname)

        if os.path.exists(mask_path):
            subject_id = fname.split(".")[0]
            tasks.append((img_path, mask_path, subject_id, img_dir, mask_dir))

    print(f"✅ 成功匹配 {len(tasks)} 对脾脏 CT 数据，准备开始并行处理...")

    all_metadata = []

    # 执行多进程
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_subject, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Cases"):
            res = future.result()
            if res:
                all_metadata.extend(res)

    # 保存 JSON 元数据
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print("-" * 40)
    print(f"🎉 全部处理完成！")
    print(f"📈 生成高质量脾脏切片总数: {len(all_metadata)}")
    print(f"📂 数据集保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()