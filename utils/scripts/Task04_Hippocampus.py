import os
import glob
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ================= 1. Configuration =================

DATASET_ROOT = r"D:\dataset\MSD\Task04_Hippocampus"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MSD_Task04_256_Expert_SR"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 20
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)
USE_25D = True

# ================= 2. Definitions =================

TASK04_ORGAN_MAP = {
    1: "anterior hippocampus",
    2: "posterior hippocampus"
}

TASK04_EXPERT_DESC = {
    "anterior hippocampus": "anterior hippocampus, the larger bulbous head region of the hippocampus",
    "posterior hippocampus": "posterior hippocampus, the narrower elongated tail region of the hippocampus"
}


# ================= 3. Core Functions =================

def normalize_mri_zscore(img_slice):
    """
    🌟 Z-Score Normalization for MRI
    """
    mask = img_slice > 0
    if np.sum(mask) == 0: return img_slice.astype(np.float32)

    pixels = img_slice[mask]
    mean = np.mean(pixels)
    std = np.std(pixels)

    if std == 0: return np.zeros_like(img_slice, dtype=np.float32)

    img_norm = (img_slice - mean) / std
    img_norm = np.clip(img_norm, -3, 3)
    img_norm = (img_norm + 3) / 6.0

    img_norm[img_slice == 0] = 0
    return img_norm


def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    """
    🌟 Unsharp Masking for Sharpening after Upscaling
    """
    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    unsharp_image = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return np.clip(unsharp_image, 0, 255).astype(np.uint8)


def get_25d_slice(vol_data, z_index):
    """
    🌟 Spatial 2.5D with Super-Resolution Prep
    """
    depth = vol_data.shape[2]
    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    # Extract and Normalize (keep as float for high-quality resize)
    slice_prev = normalize_mri_zscore(vol_data[:, :, idx_prev])
    slice_curr = normalize_mri_zscore(vol_data[:, :, idx_curr])
    slice_next = normalize_mri_zscore(vol_data[:, :, idx_next])

    return np.stack([slice_prev, slice_curr, slice_next], axis=-1)


def resize_pad_enhanced(image, target_size, is_mask=False):
    """
    🌟 High-Quality Upscaling: Lanczos4 + USM Sharpening
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    if is_mask:
        # Mask: Nearest Neighbor
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padding_value = 0
    else:
        # Image: Lanczos4 (Best for Upscaling)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Convert to uint8 for sharpening
        resized_uint8 = (resized * 255).astype(np.uint8)

        # Apply Sharpening to counter blur
        resized = apply_unsharp_mask(resized_uint8)
        padding_value = 0  # Black

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    if is_mask:
        return np.pad(resized, ((top, bottom), (left, right)), mode='constant', constant_values=padding_value)
    else:
        # Image is already (H, W, 3)
        return np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=padding_value)


def process_single_case(args):
    """ Worker Function """
    img_path, mask_path, subject_id, output_img_dir, output_mask_dir = args
    meta_list = []

    try:
        # Load NIfTI
        img_nii = nib.as_closest_canonical(nib.load(img_path))
        mask_nii = nib.as_closest_canonical(nib.load(mask_path))

        img_data = img_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.uint8)

        # Dimension fix
        if img_data.ndim == 4: img_data = img_data[:, :, :, 0]
        if mask_data.ndim == 4: mask_data = mask_data[:, :, :, 0]

        depth = img_data.shape[2]

        for z in range(depth):
            mask_slice = mask_data[:, :, z]
            if np.sum(mask_slice) == 0: continue

            # === Image Processing ===
            if USE_25D:
                img_processed = get_25d_slice(img_data, z)
                modality_str = "MRI (Spatial 2.5D)"
            else:
                slice_norm = normalize_mri_zscore(img_data[:, :, z])
                img_processed = np.stack([slice_norm] * 3, axis=-1)
                modality_str = "MRI"

            # === Resize (Lanczos) & Sharpen & Pad ===
            img_padded = resize_pad_enhanced(img_processed, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad_enhanced(mask_slice, IMG_SIZE, is_mask=True)

            file_base_id = f"msd_task04_{subject_id}_{z:03d}"

            # Save Image
            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_padded)
            rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            # Process Organs
            unique_vals = np.unique(mask_padded)
            for val in unique_vals:
                if val == 0 or val not in TASK04_ORGAN_MAP: continue

                organ_name = TASK04_ORGAN_MAP[val]
                binary_mask = (mask_padded == val).astype(np.uint8)

                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

                # Save Mask
                task_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{task_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                desc_text = TASK04_EXPERT_DESC[organ_name]

                # ✅ Standardized Metadata
                meta_list.append({
                    "img_path": rel_img,
                    "mask_path": rel_mask,
                    "modality": modality_str,  # 🌟 Attribute
                    "description": desc_text,  # 🌟 Attribute
                    "organ": organ_name,  # 🌟 Attribute
                    "raw_organ_id": int(val),
                    "source": "MSD_Task04"
                })

    except Exception as e:
        print(f"Error processing {subject_id}: {e}")
        return []

    return meta_list


# ================= 4. Main =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting MSD Task04 (Hippocampus) Preprocessing")
    print(f"✨ Strategy: Lanczos4 Upscale + USM Sharpening | Workers: {NUM_WORKERS}")

    # Find Folders
    img_folder = os.path.join(DATASET_ROOT, "imagesTr")
    lbl_folder = os.path.join(DATASET_ROOT, "labelsTr")

    if not os.path.exists(img_folder):
        # Recursive search
        for root, dirs, files in os.walk(DATASET_ROOT):
            if "imagesTr" in dirs:
                img_folder = os.path.join(root, "imagesTr")
                lbl_folder = os.path.join(root, "labelsTr")
                break

    if not os.path.exists(img_folder):
        print(f"❌ imagesTr not found in {DATASET_ROOT}")
        return

    # Build Tasks
    tasks = []
    img_files = glob.glob(os.path.join(img_folder, "*.nii*"))

    for img_path in img_files:
        fname = os.path.basename(img_path)
        mask_path = os.path.join(lbl_folder, fname)
        if os.path.exists(mask_path):
            # Clean subject ID
            sid = fname.split(".")[0]
            if sid.startswith("._"): sid = sid[2:]  # Fix macOS artifacts

            tasks.append((img_path, mask_path, sid, img_dir, mask_dir))

    print(f"✅ Found {len(tasks)} cases")

    all_metadata = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_case, task) for task in tasks]

        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            res = future.result()
            if res: all_metadata.extend(res)

    json_path = os.path.join(OUTPUT_DIR, "dataset_metadata.json")
    with open(json_path, "w") as f:
        json.dump(all_metadata, f, indent=4)

    print(f"🎉 Finished! Samples: {len(all_metadata)}")
    print(f"📂 Metadata: {json_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()