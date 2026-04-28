import os
import glob
import random
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 1. 配置区域 =================

DATASET_ROOT = r"D:\dataset\Dataset_BUSI_with_GT"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\BUSI_256_Expert_PreserveAR"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# 参数
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20  # 过滤微小噪声
INCLUDE_NORMAL = False  # 🌟 开关：是否包含正常样本 (无掩膜)

# ⚡ 多进程配置
NUM_WORKERS = 8  # 建议设置为 CPU 核心数

# ================= 2. 定义区域 =================

CLASS_MAP = {
    "benign": "benign breast tumor",
    "malignant": "malignant breast tumor",
    "normal": "normal breast tissue"
}

# 专家级描述 (BI-RADS 特征)
BUSI_EXPERT_DESC = {
    "benign": "benign breast lesion, typically oval-shaped with circumscribed margins and parallel orientation",
    "malignant": "malignant breast lesion, appearing irregular with indistinct or spiculated margins, often taller-than-wide",
    "normal": "normal breast tissue with homogeneous echotexture and no focal mass"
}


# ================= 3. 核心函数 =================

def preprocess_ultrasound(img_rgb):
    """
    🌟 超声增强流水线:
    灰度 -> 双边滤波去噪 -> 归一化 -> CLAHE
    """
    # 1. 转灰度
    if len(img_rgb.shape) == 3:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_rgb

    # 2. 去噪 (保留边缘)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)

    # 3. 鲁棒截断
    lower = np.percentile(denoised, 1.0)
    upper = np.percentile(denoised, 99.0)
    clipped = np.clip(denoised, lower, upper)

    # 4. 归一化
    if upper == lower:
        norm = np.zeros_like(clipped, dtype=np.uint8)
    else:
        norm = (clipped - lower) / (upper - lower)
        norm = (norm * 255).astype(np.uint8)

    # 5. CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # 6. 转回 3 通道
    return np.stack([enhanced] * 3, axis=-1)


def resize_pad_preserve_ar(image, target_size, is_mask=False):
    """
    🌟 保持长宽比 Resize + Padding
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 计算 Padding (居中)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    padding_value = 0

    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant', constant_values=padding_value)
    else:
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=padding_value)


def find_mask_files(image_path):
    """ 查找关联的 Mask 文件 """
    dir_name = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    name_no_ext = os.path.splitext(file_name)[0]

    search_pattern = os.path.join(dir_name, f"{name_no_ext}_mask*.png")
    return glob.glob(search_pattern)


# ================= 4. 单例处理逻辑 (Worker) =================

def process_single_case(args):
    """
    多进程 Worker 函数
    Args: (img_path, output_img_dir, output_mask_dir)
    Returns: List of dict (meta_data)
    """
    img_path, output_img_dir, output_mask_dir = args
    meta_list = []

    try:
        # 获取类别名
        class_name_raw = os.path.basename(os.path.dirname(img_path))

        # 过滤 Normal
        if class_name_raw == "normal" and not INCLUDE_NORMAL:
            return []

        organ_name = CLASS_MAP.get(class_name_raw, "breast lesion")
        desc_text = BUSI_EXPERT_DESC.get(class_name_raw, "ultrasound of breast")

        # 加载图片
        img = cv2.imread(img_path)
        if img is None: return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 加载并合并 Mask
        mask_files = find_mask_files(img_path)
        h, w = img.shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)

        if mask_files:
            for mf in mask_files:
                m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    final_mask = np.maximum(final_mask, m)

        has_object = np.max(final_mask) > 0

        # 如果不是 normal 类别且没有 mask，跳过
        if not has_object and class_name_raw != "normal":
            return []

        # === 预处理 ===
        img_processed = preprocess_ultrasound(img)

        # === Resize (保持比例) ===
        img_padded = resize_pad_preserve_ar(img_processed, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad_preserve_ar(final_mask, IMG_SIZE, is_mask=True)

        # 二值化
        mask_padded = (mask_padded > 127).astype(np.uint8)

        # 过滤微小噪声
        if has_object and np.sum(mask_padded) < MIN_PIXEL_THRESHOLD:
            return []

        # 保存文件
        file_name = os.path.basename(img_path)
        clean_name = file_name.replace(" ", "_").replace("(", "").replace(")", "").replace(".png", "")
        save_id = f"busi_{clean_name}"

        img_save_path = os.path.join(output_img_dir, f"{save_id}.npy")
        mask_save_path = os.path.join(output_mask_dir, f"{save_id}.npy")

        np.save(img_save_path, img_padded)
        np.save(mask_save_path, mask_padded)

        rel_img_path = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask_path = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ 标准元数据
        meta_list.append({
            "img_path": rel_img_path,
            "mask_path": rel_mask_path,
            "modality": "Ultrasound",  # 🌟 属性：模态
            "description": desc_text,  # 🌟 属性：详细描述
            "organ": organ_name,  # 🌟 属性：器官名
            "raw_organ_id": class_name_raw,
            "source": "BUSI"
        })

        return meta_list

    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return []


# ================= 5. 主程序 =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 开始 BUSI 多进程预处理")
    print(f"✨ 策略: 去噪 + CLAHE + 保持比例 | 进程数: {NUM_WORKERS}")

    # 收集图片路径
    image_list = []
    categories = ["benign", "malignant"]
    if INCLUDE_NORMAL: categories.append("normal")

    for sub in categories:
        search_path = os.path.join(DATASET_ROOT, sub, "*.png")
        files = glob.glob(search_path)
        # 排除 _mask 文件
        files = [f for f in files if "_mask" not in f]
        image_list.extend(files)

    print(f"✅ 找到 {len(image_list)} 张原始图像")

    all_meta_data = []
    # 准备任务参数
    tasks = [(path, img_dir, mask_dir) for path in image_list]

    # ⚡ 启动进程池
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_case, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            result = future.result()
            if result:
                all_meta_data.extend(result)

    # 保存元数据
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_meta_data, f, indent=4)

    print(f"🎉 全部完成! 共生成 {len(all_meta_data)} 个样本")
    print(f"📂 元数据保存至: {json_path}")


if __name__ == "__main__":
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 路径不存在: {DATASET_ROOT}")
    else:
        main()