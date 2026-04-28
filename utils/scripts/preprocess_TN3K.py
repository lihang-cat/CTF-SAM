import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\TN3K"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\TN3K_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 50
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. Definitions =================

ORGAN_NAME = "thyroid nodule"
RAW_ORGAN_ID = 1

# Expert Description
TN3K_EXPERT_DESC = "thyroid nodule, a discrete lesion within the thyroid gland, often hypoechoic or heterogeneous"


# ================= 3. Core Functions =================

def preprocess_ultrasound_enhancement(img_bgr):
    """
    🌟 Ultrasound Enhancement Pipeline:
    Grayscale -> Bilateral Filter (Speckle Reduction) -> CLAHE -> RGB Stack
    """
    if img_bgr.ndim == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    # Bilateral Filter: Smooths noise while keeping edges sharp
    # d=9, sigmaColor=75, sigmaSpace=75
    img_filtered = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_filtered)

    # Stack to 3 channels
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
        img_bgr = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask is None: return None

        # Enhance
        img_enhanced = preprocess_ultrasound_enhancement(img_bgr)

        # Resize & Pad
        img_padded = resize_pad(img_enhanced, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask, IMG_SIZE, is_mask=True)

        # Threshold
        binary_mask = (mask_padded > 127).astype(np.uint8)

        # Filter tiny noise
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Save
        file_name = os.path.basename(img_path)
        clean_id = os.path.splitext(file_name)[0]
        file_base_id = f"tn3k_{clean_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_id = f"{file_base_id}_thyroid_nodule"
        mask_save_path = os.path.join(mask_save_dir, f"{task_id}.npy")
        np.save(mask_save_path, binary_mask)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Ultrasound",  # 🌟 Attribute
            "description": TN3K_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "TN3K"
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

    print(f"🚀 Starting TN3K Preprocessing")
    print(f"✨ Strategy: Bilateral + CLAHE | Workers: {NUM_WORKERS}")

    tasks = []

    # Robust Folder Search
    # TN3K structure: trainval-image / trainval-mask OR just image / mask
    search_roots = [
        (os.path.join(DATASET_ROOT, "trainval-image"), os.path.join(DATASET_ROOT, "trainval-mask")),
        (os.path.join(DATASET_ROOT, "test-image"), os.path.join(DATASET_ROOT, "test-mask"))
    ]

    # Fallback to general search if specific folders missing
    if not any(os.path.exists(p[0]) for p in search_roots):
        search_roots = [(DATASET_ROOT, DATASET_ROOT)]

    for img_root, mask_root in search_roots:
        if not os.path.exists(img_root): continue

        # Find all images
        extensions = ["*.jpg", "*.png", "*.jpeg"]
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(img_root, ext)))

        for img_path in images:
            fname = os.path.basename(img_path)
            # Try matching mask in mask_root
            mask_path = os.path.join(mask_root, fname)

            # Some datasets have different extension for mask
            if not os.path.exists(mask_path):
                mask_path = os.path.splitext(mask_path)[0] + ".png"  # try png

            if os.path.exists(mask_path):
                tasks.append((img_path, mask_path, img_save_dir, mask_save_dir))

    print(f"✅ Found {len(tasks)} pairs")

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