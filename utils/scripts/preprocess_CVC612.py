import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\CVC-612"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\CVC_ClinicDB_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. Definitions =================

ORGAN_NAME = "colon polyp"
RAW_ORGAN_ID = 1

# Expert Description (Endoscopy)
CVC_EXPERT_DESC = "colon polyp, a protruding growth on the inner lining of the colon, appearing reddish and textured"


# ================= 3. Core Functions =================

def remove_specular_reflections(img_bgr):
    """
    🌟 [New] Simple Specular Reflection Removal
    Detects bright spots and inpaints them.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold to find bright spots (e.g., > 240)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Dilate mask slightly to cover edges of glare
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaint
    # Radius 3, Telea method
    img_inpainted = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    return img_inpainted


def preprocess_polyp_enhancement(img_bgr):
    """
    🌟 Endoscopy Pipeline:
    Specular Removal -> LAB CLAHE -> RGB
    """
    # 1. Remove Glare
    img_clean = remove_specular_reflections(img_bgr)

    # 2. BGR -> LAB
    lab = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)

    # 3. CLAHE on L-channel
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 4. Merge & RGB
    lab_merged = cv2.merge((l_enhanced, a, b))
    img_rgb = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)

    return img_rgb


def resize_pad(image, target_size, is_mask=False):
    """ Resize & Pad (Center) """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    padding_value = 0  # Black padding matches endoscopy field of view

    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def process_single_pair(args):
    """ Worker function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # Load
        img_bgr = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None or mask is None: return None

        # Check Mask
        binary_mask = (mask > 127).astype(np.uint8)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Preprocess
        img_rgb = preprocess_polyp_enhancement(img_bgr)

        # Resize & Pad
        img_padded = resize_pad(img_rgb, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(binary_mask, IMG_SIZE, is_mask=True)  # 0-1 mask

        # Save
        file_name = os.path.basename(img_path)
        file_id = os.path.splitext(file_name)[0]
        file_base_id = f"cvc_{file_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_file_id = f"{file_base_id}_{ORGAN_NAME.replace(' ', '_')}"
        mask_save_path = os.path.join(mask_save_dir, f"{task_file_id}.npy")
        np.save(mask_save_path, mask_padded)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Colonoscopy",  # 🌟 Attribute
            "description": CVC_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "CVC-ClinicDB"
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

    print(f"🚀 Starting CVC-ClinicDB Preprocessing")
    print(f"✨ Strategy: Specular Removal + LAB CLAHE")

    # Path setup
    # Try different casing for PNG folder
    base_dir = os.path.join(DATASET_ROOT, "PNG")
    if not os.path.exists(base_dir): base_dir = os.path.join(DATASET_ROOT, "png")

    original_dir = os.path.join(base_dir, "Original")
    gt_dir = os.path.join(base_dir, "Ground Truth")

    if not os.path.exists(original_dir):
        print(f"❌ Original dir not found: {original_dir}")
        return

    image_files = glob.glob(os.path.join(original_dir, "*.png"))
    print(f"✅ Found {len(image_files)} images")

    tasks = []
    for img_path in image_files:
        file_name = os.path.basename(img_path)
        mask_path = os.path.join(gt_dir, file_name)
        if os.path.exists(mask_path):
            tasks.append((img_path, mask_path, img_save_dir, mask_save_dir))

    valid_metadata = []

    # Start Workers
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