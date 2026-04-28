import os
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\kvasir-seg"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\Kvasir_256_Expert_SpecRemoval"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. Definitions =================

ORGAN_NAME = "colon polyp"
RAW_ORGAN_ID = 1

# Expert Description
KVASIR_EXPERT_DESC = "colon polyp, a protruding growth on the inner lining of the colon, appearing reddish and textured"


# ================= 3. Core Functions =================

def remove_specular_reflections(img_bgr):
    """
    🌟 [New] Simple Specular Reflection Removal
    Detects bright spots and inpaints them.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Dilate mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Inpaint
    img_inpainted = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
    return img_inpainted


def preprocess_polyp_enhancement(img_bgr):
    """
    🌟 Endoscopy Pipeline:
    Specular Removal -> LAB CLAHE -> RGB
    """
    # 1. Remove Glare
    img_clean = remove_specular_reflections(img_bgr)

    # 2. LAB CLAHE
    lab = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
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

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_CUBIC
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
    """ Worker function """
    img_path, mask_path, img_save_dir, mask_save_dir = args

    try:
        # Load
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return None

        # Check Mask
        if not os.path.exists(mask_path): return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Kvasir masks are usually jpg/png
        if mask is None: return None

        # Enhance
        img_rgb = preprocess_polyp_enhancement(img_bgr)

        # Resize & Pad
        img_padded = resize_pad(img_rgb, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask, IMG_SIZE, is_mask=True)

        # Threshold
        binary_mask = (mask_padded > 127).astype(np.uint8)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Save
        file_name = os.path.basename(img_path)
        clean_name = os.path.splitext(file_name)[0]
        file_base_id = f"kvasir_{clean_name}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_id = f"{file_base_id}_polyp"
        mask_save_path = os.path.join(mask_save_dir, f"{task_id}.npy")
        np.save(mask_save_path, binary_mask)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Colonoscopy",  # 🌟 Attribute
            "description": KVASIR_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "Kvasir-SEG"
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

    print(f"🚀 Starting Kvasir-SEG Preprocessing")
    print(f"✨ Strategy: Specular Removal + LAB CLAHE")

    # Smart Folder Detection
    raw_img_dir = os.path.join(DATASET_ROOT, "images")
    raw_mask_dir = os.path.join(DATASET_ROOT, "masks")

    if not os.path.exists(raw_img_dir) and os.path.exists(os.path.join(DATASET_ROOT, "image")):
        raw_img_dir = os.path.join(DATASET_ROOT, "image")

    if not os.path.exists(raw_img_dir):
        print(f"❌ Images folder not found: {raw_img_dir}")
        return

    all_images = glob.glob(os.path.join(raw_img_dir, "*.jpg")) + \
                 glob.glob(os.path.join(raw_img_dir, "*.png"))

    print(f"✅ Found {len(all_images)} images")

    tasks = []
    for img_path in all_images:
        file_name = os.path.basename(img_path)
        mask_path = os.path.join(raw_mask_dir, file_name)
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