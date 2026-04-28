import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

# 🔴 路径配置 (请修改为你的实际路径)
DATASET_ROOT = r"D:\dataset\Tufts Dental Database"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\Tufts_Teeth_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# 🔴 图像参数
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 50  # 牙齿区域通常较大，过滤掉极小的噪声

# 🔴 并行核数
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. 专家级描述定义 =================

ORGAN_NAME = "teeth"
RAW_ORGAN_ID = 1  # 二值化后前景 ID

# 专家级描述：强调模态(全景片)和特征(不透射线的钙化结构)
TUFTS_EXPERT_DESC = "teeth, the radio-opaque calcified structures arranged in the dental arches on a panoramic X-ray"


# ================= 3. 核心工具函数 =================

def preprocess_dental_enhancement(img_bgr):
    """
    🌟 Dental X-Ray Enhancement:
    Grayscale -> CLAHE -> RGB Stack
    """
    if img_bgr.ndim == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    # CLAHE
    # ClipLimit=2.0: 适度增强，过高会放大金属伪影
    # tileGridSize=(8,8): 标准网格
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # 3. 堆叠为 3 通道 (适配 CLIP 输入)
    return np.stack([img_enhanced] * 3, axis=-1)


def resize_pad(image, target_size, is_mask=False):
    """
    ⭐⭐⭐ 保持长宽比缩放 + Padding ⭐⭐⭐
    牙科全景片非常扁长，必须 Pad，否则变形严重影响分割效果
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # 插值方式：Mask用最近邻，图像用线性
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # 计算 Padding 大小 (居中填充)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    # 填充值：Mask填0，图像填0(黑色)
    val = 0 if is_mask else [0, 0, 0]

    # 边界类型
    border_type = cv2.BORDER_CONSTANT

    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, border_type, value=val)
    else:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, border_type, value=val)


def build_mask_index(root_dir):
    """
    🌟 Build a hash map for fast mask lookup
    Tufts structure: Segmentation/teeth_mask/*.jpg
    Returns: { '1': 'path/to/1.jpg', ... }
    """
    mask_index = {}
    print("⏳ Building mask index...")

    # Recursively find all JPGs in Segmentation folder
    # 这种方式比硬编码路径更鲁棒
    all_files = glob.glob(os.path.join(root_dir, "Segmentation", "**", "*.jpg"), recursive=True)

    for f in all_files:
        fname = os.path.basename(f)
        # 排除掉非 mask 文件（虽然在这个文件夹里应该都是 mask）
        # Tufts 的 mask 放在 teeth_mask 文件夹下
        if "teeth_mask" not in os.path.dirname(f) and "mask" not in fname.lower():
            continue

        # Key: filename without extension (e.g., "1")
        key = os.path.splitext(fname)[0]
        mask_index[key] = f

    print(f"✅ Indexed {len(mask_index)} masks")
    return mask_index


def process_single_pair(args):
    """ Worker Function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # 1. 读取
        img_bgr = cv2.imread(img_path)
        mask_bgr = cv2.imread(mask_path)  # Mask 也是 JPG，可能读进来是 3 通道

        if img_bgr is None or mask_bgr is None:
            return None

        # 2. 掩码处理 (JPG 压缩修复)
        # 如果 Mask 是 RGB，转灰度
        if mask_bgr.ndim == 3:
            mask_gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_bgr

        # 二值化：过滤 JPG 边缘噪声
        mask_binary = (mask_gray > 127).astype(np.uint8)

        # 3. 图像增强
        img_enhanced = preprocess_dental_enhancement(img_bgr)

        # 4. Resize & Pad
        img_padded = resize_pad(img_enhanced, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask_binary, IMG_SIZE, is_mask=True)

        # 5. 有效性检查
        if np.sum(mask_padded) < MIN_PIXEL_THRESHOLD:
            return None

        # 6. 保存
        file_name = os.path.basename(img_path)
        file_id = os.path.splitext(file_name)[0]
        file_base_id = f"tufts_{file_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_id = f"{file_base_id}_{ORGAN_NAME}"
        mask_save_path = os.path.join(mask_save_dir, f"{task_id}.npy")
        np.save(mask_save_path, mask_padded)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "X-ray",  # 🌟 Attribute
            "description": TUFTS_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "Tufts_Dental"
        }

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# ================= 4. 主流程 =================

def main():
    img_save_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_save_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    print(f"🚀 Starting Tufts Dental (Teeth) Preprocessing")
    print(f"✨ Strategy: CLAHE + Aspect Ratio Padding | Workers: {NUM_WORKERS}")

    # 1. Build Mask Index
    mask_index = build_mask_index(DATASET_ROOT)

    # 2. Find Source Images (Radiographs)
    source_root = os.path.join(DATASET_ROOT, "Radiographs")
    if not os.path.exists(source_root):
        print(f"⚠️ Radiographs folder not found, searching root...")
        source_root = DATASET_ROOT

    all_files = glob.glob(os.path.join(source_root, "**", "*.jpg"), recursive=True)
    print(f"🔍 Scanned {len(all_files)} radiographs.")

    tasks = []
    for f in all_files:
        fname = os.path.basename(f)
        parent = os.path.dirname(f).lower()

        # Skip if file is in Segmentation folder or has 'mask' in name (avoid self-matching)
        if "segmentation" in parent or "mask" in parent:
            continue

        # Key: filename without extension
        key = os.path.splitext(fname)[0]

        if key in mask_index:
            tasks.append((f, mask_index[key], img_save_dir, mask_save_dir))

    print(f"✅ Matched {len(tasks)} image-mask pairs. Starting processing...")

    valid_metadata = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_pair, t) for t in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks)):
            res = future.result()
            if res:
                valid_metadata.append(res)

    # 保存元数据
    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(valid_metadata, f, indent=4)

    print(f"🎉 Finished! Samples: {len(valid_metadata)}")
    print(f"📂 Metadata: {json_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()