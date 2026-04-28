import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\DRIVE"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\DRIVE_256_Expert_FOV"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)

# ================= 2. Definitions =================

ORGAN_NAME = "retinal vessels"
RAW_ORGAN_ID = 1

# Expert Description
DRIVE_EXPERT_DESC = "retinal vessels, the thin branching vascular network extending from the optic disc"


# ================= 3. Core Functions =================

def crop_to_fov(img_rgb, mask=None, tol=10):
    """
    🌟 [New] Auto-Crop to Field of View (FOV)
    Removes excess black borders to increase vessel resolution.
    """
    # Convert to gray to find the circle
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Threshold to find the FOV mask (pixels > tol)
    mask_fov = gray > tol

    # Find coordinates
    coords = np.argwhere(mask_fov)

    if len(coords) == 0: return img_rgb, mask  # Fallback if image is all black

    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Crop Image
    img_cropped = img_rgb[x0:x1, y0:y1]

    # Crop Mask if provided
    mask_cropped = None
    if mask is not None:
        mask_cropped = mask[x0:x1, y0:y1]

    return img_cropped, mask_cropped


def preprocess_fundus_enhancement(img_rgb):
    """
    🌟 Optimized Enhancement:
    RGB -> Green Channel CLAHE -> Merge back to RGB
    Keeps color context for CLIP while boosting vessels.
    """
    # Split
    r, g, b = cv2.split(img_rgb)

    # CLAHE on Green Channel (Vessels are darkest here)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_enhanced = clahe.apply(g)

    # Merge back (Replace G with Enhanced G)
    # Alternatively, you can copy G to all channels if you want grayscale-like
    # But keeping R and B helps distinguish Optic Disc.
    img_enhanced = cv2.merge((r, g_enhanced, b))

    return img_enhanced


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
        # Load Image (TIF)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load Mask (GIF via PIL)
        try:
            mask_pil = Image.open(mask_path)
            mask = np.array(mask_pil)
        except Exception:
            return None

        # 🌟 1. Crop to FOV (Critical for resolution)
        img_cropped, mask_cropped = crop_to_fov(img_rgb, mask)

        # 2. Enhance
        img_enhanced = preprocess_fundus_enhancement(img_cropped)

        # 3. Resize & Pad
        img_padded = resize_pad(img_enhanced, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(mask_cropped, IMG_SIZE, is_mask=True)

        # 4. Threshold & Filter
        binary_mask = (mask_padded > 127).astype(np.uint8)
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Save
        file_name = os.path.basename(img_path)
        subject_id = file_name.split('_')[0]
        file_base_id = f"drive_{subject_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        task_file_id = f"{file_base_id}_vessels"
        mask_save_path = os.path.join(mask_save_dir, f"{task_file_id}.npy")
        np.save(mask_save_path, binary_mask)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Fundus Photography",  # 🌟 Attribute
            "description": DRIVE_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "DRIVE"
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

    print(f"🚀 Starting DRIVE Preprocessing")
    print(f"✨ Strategy: FOV Crop + Green Channel CLAHE")

    tasks = []
    # DRIVE structure: training/images, test/images
    subsets = ["training", "test"]

    for subset in subsets:
        subset_root = os.path.join(DATASET_ROOT, subset)
        if not os.path.exists(subset_root): continue

        img_folder = os.path.join(subset_root, "images")
        mask_folder = os.path.join(subset_root, "1st_manual")

        # DRIVE images are .tif
        img_files = glob.glob(os.path.join(img_folder, "*.tif"))

        for img_path in img_files:
            # File matching: 21_training.tif -> 21_manual1.gif
            file_name = os.path.basename(img_path)
            file_id = file_name.split('_')[0]

            mask_name = f"{file_id}_manual1.gif"
            mask_path = os.path.join(mask_folder, mask_name)

            if os.path.exists(mask_path):
                tasks.append((img_path, mask_path, img_save_dir, mask_save_dir))

    print(f"✅ Found {len(tasks)} image pairs")

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