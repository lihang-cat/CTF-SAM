import os
import glob
import json
import numpy as np
import cv2
import pandas as pd
import pydicom
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\siim"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\SIIM_256_Expert_Std"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 30
NUM_WORKERS = 8

# ================= 2. Definitions =================

ORGAN_NAME = "pneumothorax"
RAW_ORGAN_ID = 1

# Expert Description
SIIM_EXPERT_DESC = "pneumothorax, a dark area in the pleural space where the lung has collapsed, devoid of lung markings"


# ================= 3. Core Functions =================

def rle2mask(mask_rle, shape):
    """
    🌟 Fast Vectorized RLE Decoding
    SIIM RLE is run-length encoded, 1-based, column-major
    """
    if pd.isna(mask_rle) or str(mask_rle).strip() == '-1':
        return np.zeros(shape, dtype=np.uint8)

    s = np.array(str(mask_rle).split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # Vectorized assignment (much faster than loop)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape[1], shape[0]).T


def apply_dicom_window(ds, img):
    """
    🌟 Standard DICOM Windowing
    """
    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        # Handle multi-value window centers (use first)
        wc = ds.WindowCenter if isinstance(ds.WindowCenter, float) else ds.WindowCenter[0]
        ww = ds.WindowWidth if isinstance(ds.WindowWidth, float) else ds.WindowWidth[0]

        lower = wc - ww / 2.0
        upper = wc + ww / 2.0

        img = np.clip(img, lower, upper)
        # Normalize to 0-255
        if upper == lower: return np.zeros_like(img)
        img = (img - lower) / (upper - lower) * 255.0
    else:
        # Fallback: Robust Min-Max (1% - 99%)
        lower = np.percentile(img, 1.0)
        upper = np.percentile(img, 99.0)
        img = np.clip(img, lower, upper)
        if upper == lower: return np.zeros_like(img)
        img = (img - lower) / (upper - lower) * 255.0

    return img


def preprocess_dicom_enhancement(ds):
    """
    🌟 X-Ray Enhancement Pipeline:
    Load -> Photometric Correct -> Windowing -> CLAHE -> RGB
    """
    img = ds.pixel_array.astype(float)

    # 1. Photometric Interpretation (Fix Inverted X-rays)
    if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img

    # 2. Apply Windowing
    img = apply_dicom_window(ds, img)
    img_uint8 = img.astype(np.uint8)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_uint8)

    # 4. Stack to 3 channels
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


def process_single_sample(args):
    """ Worker Function """
    dcm_path, rle_codes, img_save_dir, mask_save_dir = args

    try:
        # Load DICOM
        ds = pydicom.dcmread(dcm_path)
        orig_h, orig_w = ds.pixel_array.shape

        # Enhance
        img_rgb = preprocess_dicom_enhancement(ds)

        # Merge Masks (Handle multiple annotations)
        final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if isinstance(rle_codes, str): rle_codes = [rle_codes]

        for rle in rle_codes:
            mask = rle2mask(rle, (orig_h, orig_w))
            final_mask = np.maximum(final_mask, mask)

        # Resize & Pad
        img_padded = resize_pad(img_rgb, IMG_SIZE, is_mask=False)
        mask_padded = resize_pad(final_mask, IMG_SIZE, is_mask=True)

        # Threshold
        binary_mask = (mask_padded > 0).astype(np.uint8)

        # Filter empty or tiny masks
        if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: return None

        # Save
        file_id = os.path.splitext(os.path.basename(dcm_path))[0]
        file_base_id = f"siim_{file_id}"

        img_save_path = os.path.join(img_save_dir, f"{file_base_id}.npy")
        np.save(img_save_path, img_padded)

        mask_save_path = os.path.join(mask_save_dir, f"{file_base_id}_pneumo.npy")
        np.save(mask_save_path, binary_mask)

        rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")
        rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

        # ✅ Standardized Metadata
        return {
            "img_path": rel_img,
            "mask_path": rel_mask,
            "modality": "Chest X-Ray",  # 🌟 Attribute
            "description": SIIM_EXPERT_DESC,  # 🌟 Attribute
            "organ": ORGAN_NAME,  # 🌟 Attribute
            "raw_organ_id": RAW_ORGAN_ID,
            "source": "SIIM_Pneumothorax"
        }

    except Exception as e:
        print(f"Error processing {dcm_path}: {e}")
        return None


# ================= 4. Main =================

def main():
    img_save_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_save_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    print(f"🚀 Starting SIIM Pneumothorax Preprocessing")
    print(f"✨ Strategy: Windowing + CLAHE | Workers: {NUM_WORKERS}")

    # 1. Read CSV
    csv_file = "train-rle.csv"
    if not os.path.exists(os.path.join(DATASET_ROOT, csv_file)):
        # Try stage 2
        csv_file = "stage_2_train.csv"

    csv_path = os.path.join(DATASET_ROOT, csv_file)
    if not os.path.exists(csv_path):
        print(f"❌ CSV not found in {DATASET_ROOT}")
        return

    print(f"⏳ Reading CSV: {csv_file}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Create RLE Map: ImageId -> List of RLEs
    rle_col = 'EncodedPixels' if 'EncodedPixels' in df.columns else df.columns[1]
    # Filter out -1 (No Pneumothorax)
    df_valid = df[df[rle_col] != '-1']

    if df_valid.empty:
        print("❌ No valid pneumothorax samples found in CSV (all are -1?)")
        # If you want to include negatives, remove the filter above
        return

    rle_map = df_valid.groupby('ImageId')[rle_col].apply(list).to_dict()
    print(f"✅ Found {len(rle_map)} positive cases")

    # 2. Find DICOMs
    print(f"🔍 Scanning DICOMs...")
    all_dcms = glob.glob(os.path.join(DATASET_ROOT, "**", "*.dcm"), recursive=True)

    # 3. Match and Build Tasks
    tasks = []
    # Create a quick lookup for DICOM paths
    dcm_lookup = {os.path.splitext(os.path.basename(p))[0]: p for p in all_dcms}

    for img_id, rle_list in rle_map.items():
        if img_id in dcm_lookup:
            tasks.append((dcm_lookup[img_id], rle_list, img_save_dir, mask_save_dir))

    print(f"✅ Matched {len(tasks)} tasks")

    valid_metadata = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_sample, task) for task in tasks]

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