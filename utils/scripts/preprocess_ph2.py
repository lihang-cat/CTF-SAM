import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

# ⚠️ 适配 PH2 的路径 (根据你之前的截图)
DATASET_ROOT = r"D:\Dataset\PH2Dataset\PH2 Dataset images"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\PH2_256_Expert_DullRazor"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# ================= 2. Definitions =================

ORGAN_NAME = "skin lesion"
RAW_ORGAN_ID = 1

# Expert Description (Dermatology)
# PH2 也是皮肤病灶，沿用专业的描述
PH2_EXPERT_DESC = "skin lesion, a pigmented area on the skin with irregular borders, asymmetry, or color variation"


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

    # Avoid division by zero
    if mean_r == 0: mean_r = 1
    if mean_g == 0: mean_g = 1
    if mean_b == 0: mean_b = 1

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

    padding_value = 0

    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_value)
    else:
        # Use simple black padding
        return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])


def process_single_pair(args):
    """ Worker function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # Load (PH2 images are .bmp, cv2 handles this automatically)
        img_bgr = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img_bgr is None:
            print(f"❌ Failed to load image: {img_path}")
            return None
        if mask is None:
            print(f"❌ Failed to load mask: {mask_path}")
            return None

        # Check Mask
        binary_mask = (mask > 127).astype(np.uint8)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Preprocess (Enhancement Pipeline)
        img_rgb = preprocess_dermoscopy_enhancement(img_bgr)

        # Resize & Pad
        img_padded = resize_pad(img_rgb, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(binary_mask, IMG_SIZE, is_mask=True)  # 0-1 mask

        # Save
        # PH2 ID format: IMD002
        file_name = os.path.basename(img_path)
        clean_id = os.path.splitext(file_name)[0]
        file_base_id = f"ph2_{clean_id}"

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
            "description": PH2_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "PH2"
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

    print(f"🚀 Starting PH2 Dataset Preprocessing")
    print(f"✨ Strategy: DullRazor (Hair Removal) + White Balance + CLAHE")
    print(f"📂 Root: {DATASET_ROOT}")

    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Root dir not found: {DATASET_ROOT}")
        return

    # 🌟 Modified Logic for PH2 Nested Structure
    # Structure: Root / IMDxxx / IMDxxx_Dermoscopic_Image / IMDxxx.bmp

    tasks = []

    # Get all Case IDs (folders starting with IMD)
    case_dirs = [d for d in os.listdir(DATASET_ROOT) if
                 os.path.isdir(os.path.join(DATASET_ROOT, d)) and d.startswith("IMD")]
    case_dirs.sort()

    print(f"🔍 Found {len(case_dirs)} case folders (IMDxxx)...")

    for case_id in case_dirs:
        case_root = os.path.join(DATASET_ROOT, case_id)

        # Construct Paths
        # Image: IMDxxx_Dermoscopic_Image/IMDxxx.bmp
        img_sub_dir = os.path.join(case_root, f"{case_id}_Dermoscopic_Image")
        img_path = os.path.join(img_sub_dir, f"{case_id}.bmp")

        # Mask: IMDxxx_lesion/IMDxxx_lesion.bmp
        mask_sub_dir = os.path.join(case_root, f"{case_id}_lesion")
        mask_path = os.path.join(mask_sub_dir, f"{case_id}_lesion.bmp")

        if os.path.exists(img_path) and os.path.exists(mask_path):
            tasks.append((img_path, mask_path, img_save_dir, mask_save_dir))
        else:
            # Silent skip or minimal log if needed
            pass

    print(f"✅ Prepared {len(tasks)} tasks.")

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