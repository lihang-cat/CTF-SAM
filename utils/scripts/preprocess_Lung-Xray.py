import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================


DATASET_ROOT = r"D:\Dataset\Montgomery_X_ray"

OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\Montgomery_ChestXray_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# ================= 2. Definitions =================

XRAY_ORGAN_MAP = {
    1: "lungs"
}

XRAY_EXPERT_DESC = {
    "lungs": "lungs, the radiolucent fields in the chest cavity appearing dark on X-ray, bounded by ribs and diaphragm"
}


# ================= 3. Core Functions =================

def preprocess_xray_enhancement(img_bgr):
    """
    🌟 X-Ray Enhancement:
    Grayscale -> CLAHE -> RGB Stack
    """
    if img_bgr.ndim == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    # CLAHE
    # ClipLimit=2.0 is standard.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_gray)

    # Stack to 3 channels for CLIP
    return np.stack([img_enhanced] * 3, axis=-1)


def resize_pad(image, target_size, is_mask=False):
    """ Resize & Pad (Center) """
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
        # X-ray background is usually black (0)
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def build_mask_index(root_dir):
    """
    🌟 Build a hash map for fast mask lookup
    Returns: { 'CHNCXR_0001_0': 'path/to/mask.png', ... }
    """
    mask_index = {}
    print("⏳ Building mask index...")

    # Recursively find all PNGs
    all_files = glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)

    for f in all_files:
        fname = os.path.basename(f)
        parent = os.path.dirname(f).lower()

        # Identify masks based on folder name or filename suffix
        is_mask = False
        key = None

        # Case 1: Shenzhen Style (_mask.png)
        if fname.endswith("_mask.png"):
            is_mask = True
            key = fname.replace("_mask.png", "")

        # Case 2: Montgomery Style (mask in 'mask' folder)
        elif "mask" in parent or "manual" in parent:
            is_mask = True
            key = os.path.splitext(fname)[0]

        if is_mask and key:
            mask_index[key] = f

    print(f"✅ Indexed {len(mask_index)} masks")
    return mask_index


def process_single_pair(args):
    """ Worker Function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # Load
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None: return None

        # Enhance
        img_enhanced = preprocess_xray_enhancement(img)

        # Resize & Pad
        img_padded = resize_pad(img_enhanced, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask, IMG_SIZE, is_mask=True)

        # Threshold
        binary_mask = (mask_padded > 127).astype(np.uint8)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Save
        file_name = os.path.basename(img_path).replace(".png", "")
        file_base_id = f"xray_{file_name}"
        organ_name = "lungs"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_id = f"{file_base_id}_{organ_name}"
        mask_save_path = os.path.join(mask_save_dir, f"{task_id}.npy")
        np.save(mask_save_path, binary_mask)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Chest X-Ray",  # 🌟 Attribute
            "description": XRAY_EXPERT_DESC["lungs"],  # 🌟 Attribute
            "organ": organ_name,  # 🌟 Attribute
            "raw_organ_id": 1,
            "source": "ChestXray"
        }

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


# ================= 4. Main =================

def main():
    img_save_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_save_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    print(f"🚀 Starting Chest X-ray Preprocessing")
    print(f"✨ Strategy: CLAHE + Fast Indexing | Workers: {NUM_WORKERS}")

    # 1. Build Mask Index (O(1) lookup)
    mask_index = build_mask_index(DATASET_ROOT)

    # 2. Find Source Images
    all_files = glob.glob(os.path.join(DATASET_ROOT, "**", "*.png"), recursive=True)
    tasks = []

    for f in all_files:
        fname = os.path.basename(f)
        parent = os.path.dirname(f).lower()

        # Skip if file is a mask itself
        if "_mask" in fname or "mask" in parent or "manual" in parent:
            continue

        # Try to find corresponding mask
        # Key: filename without extension
        key = os.path.splitext(fname)[0]

        if key in mask_index:
            tasks.append((f, mask_index[key], img_save_dir, mask_save_dir))

    print(f"✅ Matched {len(tasks)} image-mask pairs")

    valid_metadata = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_pair, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res: valid_metadata.append(res)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(valid_metadata, f, indent=4)

    print(f"🎉 Finished! Samples: {len(valid_metadata)}")
    print(f"📂 Metadata: {json_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()