import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\ultrasound-nerve-segmentation"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\Nerve_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. Definitions =================

ORGAN_NAME = "brachial plexus"
RAW_ORGAN_ID = 1

# Expert Description
NERVE_EXPERT_DESC = "brachial plexus, a honeycomb-like cluster of dark nerve circles within the neck muscle"


# ================= 3. Core Functions =================

def preprocess_ultrasound_enhancement(img_bgr):
    """
    🌟 Ultrasound Enhancement:
    Grayscale -> Bilateral Filter (Denoise) -> CLAHE (Contrast) -> RGB Stack
    """
    if img_bgr.ndim == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    # Bilateral Filter: excellent for preserving edges while removing speckle
    # d=9, sigmaColor=75, sigmaSpace=75 are robust defaults
    img_filtered = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_filtered)

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
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def process_single_pair(args):
    """ Worker Function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # Load
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None: return None

        # Enhance
        img_enhanced = preprocess_ultrasound_enhancement(img)

        # Resize & Pad
        img_padded = resize_pad(img_enhanced, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask, IMG_SIZE, is_mask=True)

        # Threshold
        binary_mask = (mask_padded > 127).astype(np.uint8)

        # Filter empty masks (Critical for this dataset)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Save
        file_name = os.path.basename(img_path)
        clean_id = file_name.replace(".tif", "")
        file_base_id = f"nerve_{clean_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_id = f"{file_base_id}_nerve"
        mask_save_path = os.path.join(mask_save_dir, f"{task_id}.npy")
        np.save(mask_save_path, binary_mask)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Ultrasound",  # 🌟 Attribute
            "description": NERVE_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "Nerve_US"
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

    print(f"🚀 Starting Nerve Ultrasound Preprocessing")
    print(f"✨ Strategy: Bilateral + CLAHE | Workers: {NUM_WORKERS}")

    # Check path
    train_dir = os.path.join(DATASET_ROOT, "train")
    if not os.path.exists(train_dir): train_dir = DATASET_ROOT  # Compatibility

    mask_files = glob.glob(os.path.join(train_dir, "*_mask.tif"))
    if not mask_files:
        print(f"❌ No masks found in {train_dir}")
        return

    print(f"✅ Found {len(mask_files)} masks")

    # Build Tasks
    tasks = []
    for mask_path in mask_files:
        img_path = mask_path.replace("_mask.tif", ".tif")
        if os.path.exists(img_path):
            tasks.append((img_path, mask_path, img_save_dir, mask_save_dir))

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