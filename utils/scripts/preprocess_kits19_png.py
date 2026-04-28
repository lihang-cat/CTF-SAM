import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. 配置区域 =================

# 🔴 路径配置
DATASET_ROOT = r"D:\dataset\KiTS19\kits19_png"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\KiTS19_256_Expert_2D"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# 参数
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30  # 过滤极小病灶
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. 定义区域 =================

KITS_ORGAN_MAP = {
    1: "kidney",
    2: "kidney tumor"
}

KITS_EXPERT_DESC = {
    "kidney": "kidney, a bean-shaped organ in the retroperitoneal space",
    "kidney tumor": "kidney tumor, an irregular mass arising from the renal parenchyma"
}


# ================= 3. 核心函数 =================

def preprocess_2d_enhancement(img_gray):
    """
    ⭐⭐⭐ 纯 2D 增强策略 ⭐⭐⭐
    1. CLAHE: 增强局部对比度 (模拟 CT 窗宽窗位调整)
    2. Stack: 复制为 3 通道适配模型
    """
    if img_gray is None: return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # 限制对比度自适应直方图均衡
    # ClipLimit=2.0 是医学图像的经验值，太高会引入噪声
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # 转为 3 通道 RGB (灰度复制)
    return np.stack([img_enhanced] * 3, axis=-1)


def resize_pad(image, target_size, is_mask=False):
    """ 保持长宽比缩放 + 居中 Padding """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    padding_value = 0
    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        # 图像如果是 3 通道
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def fix_mask_values(mask):
    """
    ⭐⭐⭐ 关键修复 ⭐⭐⭐
    PNG 保存 Mask 时，像素值可能从 0/1/2 变成 0/128/255。
    这里将其强制映射回 0, 1, 2
    """
    unique = np.unique(mask)
    # 如果已经是标准的 0, 1, 2，直接返回
    if np.max(unique) <= 2: return mask

    new_mask = np.zeros_like(mask)
    sorted_unique = sorted(unique)  # e.g., [0, 127, 255]

    # 0 -> BG
    # 中间值 -> Kidney (1)
    # 最大值 -> Tumor (2)
    for i, val in enumerate(sorted_unique):
        if val == 0: continue
        if i == 1: new_mask[mask == val] = 1  # Kidney
        if i == 2: new_mask[mask == val] = 2  # Tumor

    return new_mask


# ================= 4. 单例处理逻辑 (Worker) =================

def process_single_case(case_folder, output_img_dir, output_mask_dir):
    """ 处理单个 Case 文件夹 """
    meta_list = []
    case_id = os.path.basename(case_folder)  # case_00000

    img_folder = os.path.join(case_folder, "imaging")
    # 兼容两种命名: segmentation 或 masks
    mask_folder = os.path.join(case_folder, "segmentation")
    if not os.path.exists(mask_folder): mask_folder = os.path.join(case_folder, "masks")

    if not os.path.exists(img_folder) or not os.path.exists(mask_folder): return []

    # 获取该 Case 下所有 PNG
    img_files = glob.glob(os.path.join(img_folder, "*.png"))

    for img_path in img_files:
        fname = os.path.basename(img_path)
        mask_path = os.path.join(mask_folder, fname)

        if not os.path.exists(mask_path): continue

        # 1. 读取 Mask (先读 Mask，如果空的就跳过，省时间)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None: continue

        # 修复 Mask 值
        mask_fixed = fix_mask_values(mask_raw)
        if np.sum(mask_fixed) == 0: continue  # 跳过背景切片

        # 2. 读取 Image
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: continue

        # === 预处理 (CLAHE + RGB Stack) ===
        img_processed = preprocess_2d_enhancement(img_gray)

        # === Resize & Pad ===
        img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask_fixed, IMG_SIZE, is_mask=True)

        # 3. 保存文件
        # ID: kits19_case_00000_slice_005
        file_base_id = f"kits19_{case_id}_{fname.replace('.png', '')}"

        img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)
        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

        # 4. 分离器官 (Kidney / Tumor)
        unique_vals = np.unique(mask_padded)
        for val in unique_vals:
            if val == 0 or val not in KITS_ORGAN_MAP: continue

            organ_name = KITS_ORGAN_MAP[val]
            binary_mask = (mask_padded == val).astype(np.uint8)

            if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

            # 保存 Mask
            task_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
            mask_save_path = os.path.join(output_mask_dir, f"{task_id}.npy")
            np.save(mask_save_path, binary_mask)
            rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

            desc_text = KITS_EXPERT_DESC[organ_name]

            # ✅ 标准化元数据
            meta_list.append({
                "img_path": rel_img,
                "mask_path": rel_mask,
                "modality": "CT",  # 🌟 模态
                "description": desc_text,  # 🌟 描述
                "organ": organ_name,  # 🌟 器官
                "raw_organ_id": int(val),
                "source": "KiTS19"
            })

    return meta_list


# ================= 5. 主程序 =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 开始 KiTS19 预处理 (纯 2D 模式)")
    print(f"✨ 策略: CLAHE 增强 + 自动修复 Mask 值 | 进程数: {NUM_WORKERS}")

    # 查找所有 case 文件夹
    case_folders = glob.glob(os.path.join(DATASET_ROOT, "case_*"))
    if not case_folders:
        print(f"❌ 未找到 case_* 文件夹，请检查路径: {DATASET_ROOT}")
        return

    print(f"✅ 找到 {len(case_folders)} 个病例")

    # 构建任务
    tasks = [(f, img_dir, mask_dir) for f in case_folders]
    all_metadata = []

    # 启动多进程
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # 包装任务参数
        futures = [executor.submit(process_single_case, *task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing Cases"):
            res = future.result()
            if res: all_metadata.extend(res)

    # 保存元数据
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"🎉 全部完成! 生成样本数: {len(all_metadata)}")
    print(f"📂 元数据: {json_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()