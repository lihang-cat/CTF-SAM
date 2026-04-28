import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\ISIC 2018"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\ISIC_256_Expert_DullRazor"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# Input/GT Folders
INPUT_DIR_NAME = "ISIC2018_Task1-2_Training_Input"
GT_DIR_NAME = "ISIC2018_Task1_Training_GroundTruth"

# ================= 2. Definitions =================

ORGAN_NAME = "skin lesion"
RAW_ORGAN_ID = 1

# Expert Description (Dermatology)
ISIC_EXPERT_DESC = "skin lesion, a pigmented area on the skin with irregular borders, asymmetry, or color variation"


# ================= 3. Core Functions =================

def dull_razor_hair_removal(img_rgb):
    """
    🌟 [SOTA] DullRazor Algorithm for Hair Removal
    1. Gray -> BlackHat -> Threshold -> Hair Mask
    2. Inpaint the hair mask
    """
    # Convert to Gray
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # BlackHat transform (Original - Closed) to find dark hair on light skin
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # Thresholding to create hair mask
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint
    img_clean = cv2.inpaint(img_rgb, hair_mask, 3, cv2.INPAINT_TELEA)

    return img_clean


def gray_world_white_balance(img_rgb):
    """
    🌟 [SOTA] Gray World Assumption for Color Constancy
    Normalizes color temperature across different clinics.
    """
    result = img_rgb.transpose(2, 0, 1).astype(np.float32)  # (3, H, W)

    # Mean of each channel
    mean_r = np.mean(result[0])
    mean_g = np.mean(result[1])
    mean_b = np.mean(result[2])

    # Average gray
    mean_gray = (mean_r + mean_g + mean_b) / 3.0

    # Scale channels
    result[0] = np.minimum(result[0] * (mean_gray / mean_r), 255)
    result[1] = np.minimum(result[1] * (mean_gray / mean_g), 255)
    result[2] = np.minimum(result[2] * (mean_gray / mean_b), 255)

    return result.transpose(1, 2, 0).astype(np.uint8)


def preprocess_dermoscopy_enhancement(img_bgr):
    """
    🌟 Full Pipeline:
    BGR -> RGB -> DullRazor (Hair Removal) -> White Balance -> LAB CLAHE
    """
    # 1. BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Hair Removal
    img_hairless = dull_razor_hair_removal(img_rgb)

    # 3. White Balance
    img_wb = gray_world_white_balance(img_hairless)

    # 4. Local Contrast Enhancement (LAB CLAHE)
    lab = cv2.cvtColor(img_wb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_merged = cv2.merge((l_enhanced, a, b))

    return cv2.cvtColor(lab_merged, cv2.COLOR_LAB2RGB)


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

    padding_value = 0  # Black padding matches vignette

    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        # Use simple black padding, or reflection padding if preferred
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
        img_rgb = preprocess_dermoscopy_enhancement(img_bgr)

        # Resize & Pad
        img_padded = resize_pad(img_rgb, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(binary_mask, IMG_SIZE, is_mask=True)  # 0-1 mask

        # Save
        file_name = os.path.basename(img_path)
        clean_id = os.path.splitext(file_name)[0]
        file_base_id = f"isic_{clean_id}"

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
            "modality": "Dermoscopy",  # 🌟 Attribute
            "description": ISIC_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "ISIC2018"
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

    print(f"🚀 Starting ISIC 2018 Preprocessing")
    print(f"✨ Strategy: DullRazor (Hair Removal) + White Balance + CLAHE")

    raw_img_dir = os.path.join(DATASET_ROOT, INPUT_DIR_NAME)
    raw_gt_dir = os.path.join(DATASET_ROOT, GT_DIR_NAME)

    if not os.path.exists(raw_img_dir):
        print(f"❌ Input dir not found: {raw_img_dir}")
        return

    # Find Images
    # ISIC 2018 images are .jpg
    img_files = glob.glob(os.path.join(raw_img_dir, "*.jpg"))
    print(f"✅ Found {len(img_files)} images")

    tasks = []
    for img_path in img_files:
        file_name = os.path.basename(img_path)
        img_id = os.path.splitext(file_name)[0]

        # Match Mask: ISIC_0000000_segmentation.png
        mask_name = f"{img_id}_segmentation.png"
        mask_path = os.path.join(raw_gt_dir, mask_name)

        if os.path.exists(mask_path):
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