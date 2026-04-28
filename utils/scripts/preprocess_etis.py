import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

# 🔴 你的 ETIS 数据集路径
DATASET_ROOT = r"D:\dataset\ETIS-LaribPolypDB"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\ETIS_Larib_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30  # ETIS里有些息肉极小，阈值可设为20
NUM_WORKERS = 8

# ================= 2. Definitions =================

ORGAN_NAME = "colon polyp"
RAW_ORGAN_ID = 1

# Expert Description (Endoscopy)
ETIS_EXPERT_DESC = "colon polyp, a protruding growth on the inner lining of the colon, appearing reddish and textured under endoscopy"

# ================= 3. Core Functions =================

def remove_specular_reflections(img_bgr):
    """ Detects bright spots (glare) and inpaints them. """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Radius 3, Telea method
    img_inpainted = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    return img_inpainted

def preprocess_polyp_enhancement(img_bgr):
    """ Endoscopy Pipeline: Specular Removal -> LAB CLAHE -> RGB """
    img_clean = remove_specular_reflections(img_bgr)
    lab = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)

def resize_pad(image, target_size, is_mask=False):
    """ Resize & Pad (Center) using OpenCV """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    padding_value = 0
    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def process_single_pair(args):
    """ Worker function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # 1. Load Original Images
        # 注意 ETIS 原图分辨率极大 (1225x996)，这一步加载需要一点时间
        img_bgr = cv2.imread(img_path)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask_gray is None: return None

        # 🌟 2. Resize FIRST (提速核心：极大降低内窥镜高清图的处理耗时)
        img_padded = resize_pad(img_bgr, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask_gray, IMG_SIZE, is_mask=True)

        # 🌟 3. Check Threshold AFTER Resize (防止产生空 Mask)
        binary_mask = (mask_padded > 127).astype(np.uint8)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # 🌟 4. Preprocess / Enhance (在 256x256 下极速运行反光消除和CLAHE)
        img_rgb_enhanced = preprocess_polyp_enhancement(img_padded)

        # 5. Save
        file_name = os.path.basename(img_path)
        file_id = os.path.splitext(file_name)[0]
        file_base_id = f"etis_{file_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_rgb_enhanced)  # 保存增强后的 RGB 图

        task_file_id = f"{file_base_id}_{ORGAN_NAME.replace(' ', '_')}"
        mask_save_path = os.path.join(mask_save_dir, f"{task_file_id}.npy")
        np.save(mask_save_path, binary_mask)  # 严格保存 0-1 二值掩码

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Colonoscopy",
            "description": ETIS_EXPERT_DESC,
            "organ": ORGAN_NAME,
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "ETIS-Larib"
        }

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(img_path)}: {e}")
        return None

# ================= 4. Main =================

def main():
    img_save_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_save_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    print(f"🚀 Starting ETIS-Larib Preprocessing (A-Type Test Dataset)")
    print(f"✨ Strategy: Fast Specular Removal + LAB CLAHE")

    # Path setup
    images_dir = os.path.join(DATASET_ROOT, "images")
    masks_dir = os.path.join(DATASET_ROOT, "masks")

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"❌ 目录未找到，请检查 DATASET_ROOT: {DATASET_ROOT}")
        return

    # 扫描图像 (支持 png/jpg)
    image_files = glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg"))
    print(f"✅ Found {len(image_files)} images in ETIS")

    tasks = []
    for img_path in image_files:
        file_name = os.path.basename(img_path)
        # 尝试匹配同名 png 或 jpg 掩码
        mask_path = os.path.join(masks_dir, file_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, os.path.splitext(file_name)[0] + ".png")

        if os.path.exists(mask_path):
            tasks.append((img_path, mask_path, img_save_dir, mask_save_dir))
        else:
            print(f"⚠️ Missing Mask for: {file_name}")

    valid_metadata = []

    # Start Workers
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_pair, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing ETIS"):
            res = future.result()
            if res: valid_metadata.append(res)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(valid_metadata, f, indent=4)

    print(f"🎉 Finished! Successful Samples: {len(valid_metadata)}")
    print(f"📂 Metadata saved: {json_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()