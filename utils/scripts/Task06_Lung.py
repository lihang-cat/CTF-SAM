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

DATASET_ROOT = r"D:\dataset\MSD\Task06_Lung"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\MSD_Task06_256_Expert_2.5D"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

# Parameters
IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 25
NUM_WORKERS = 8
USE_25D = True

# ================= 2. Definitions =================

TASK06_ORGAN_MAP = {
    1: "lung tumor"
}

# 🌟 针对 CLIP 优化的医学影像 Prompt 模板
TASK06_EXPERT_DESC = {
    "lung tumor": "A pseudo-RGB CT slice of the lungs. The red channel displays the soft tissue window, the green channel shows the lung window, and the blue channel highlights the bone window. The mask segments a lung tumor, appearing as a focal nodule or mass."
}


# ================= 3. Core Functions =================

def apply_window(img, center, width):
    """
    Standard CT Windowing
    """
    lower = center - width / 2.0
    upper = center + width / 2.0
    img_clamped = np.clip(img, lower, upper)
    if upper == lower: return np.zeros_like(img)
    return (img_clamped - lower) / (upper - lower)


def get_25d_slice(vol_data, z_index):
    """
    Spatial 2.5D: [z-1, z, z+1] with Lung Window
    """
    depth = vol_data.shape[2]
    W_CENTER, W_WIDTH = -600, 1500

    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    slice_prev = apply_window(vol_data[:, :, idx_prev], W_CENTER, W_WIDTH)
    slice_curr = apply_window(vol_data[:, :, idx_curr], W_CENTER, W_WIDTH)
    slice_next = apply_window(vol_data[:, :, idx_next], W_CENTER, W_WIDTH)

    img_merged = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
    return (img_merged * 255).astype(np.uint8)


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

        if img_data.ndim == 4: img_data = img_data[:, :, :, 0]
        if mask_data.ndim == 4: mask_data = mask_data[:, :, :, 0]

        depth = img_data.shape[2]

        # Pre-calculate valid slices to save time
        has_tumor_z = np.any(mask_data > 0, axis=(0, 1))
        valid_indices = np.where(has_tumor_z)[0]

        for z in valid_indices:
            mask_slice = mask_data[:, :, z]

            # === Image Processing ===
            if USE_25D:
                img_processed = get_25d_slice(img_data, z)
                modality_str = "CT_2.5D"
            else:
                # 🌟 2D Multi-Window Pseudo-RGB Fusion (CLIP/SAM Optimized)
                slice_curr = img_data[:, :, z]

                # R: 软组织窗 (WW: 400, WL: 40) - 凸显肿瘤实性成分、纵隔和血管
                ch_r = apply_window(slice_curr, center=40, width=400)
                # G: 肺窗 (WW: 1500, WL: -600) - 凸显肺实质、毛玻璃结节边缘
                ch_g = apply_window(slice_curr, center=-600, width=1500)
                # B: 骨窗/高对比窗 (WW: 1000, WL: 400) - 剥离肋骨干扰，凸显钙化灶
                ch_b = apply_window(slice_curr, center=400, width=1000)

                img_processed = np.stack([ch_r, ch_g, ch_b], axis=-1)
                img_processed = (img_processed * 255).astype(np.uint8)
                modality_str = "CT_MultiWindow_PseudoRGB"

            # === Resize & Pad ===
            img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
            mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

            file_base_id = f"msd_task06_{subject_id}_{z:03d}"

            # Save Image
            img_save_path = os.path.join(output_img_dir, f"{file_base_id}.npy")
            np.save(img_save_path, img_padded)
            rel_img = os.path.relpath(img_save_path, PROJECT_ROOT).replace("\\", "/")

            # Process Organs
            unique_vals = np.unique(mask_padded)
            for val in unique_vals:
                if val == 0 or val not in TASK06_ORGAN_MAP: continue

                organ_name = TASK06_ORGAN_MAP[val]
                binary_mask = (mask_padded == val).astype(np.uint8)

                if np.sum(binary_mask) < MIN_PIXEL_THRESHOLD: continue

                # Save Mask
                task_id = f"{file_base_id}_{organ_name.replace(' ', '_')}"
                mask_save_path = os.path.join(output_mask_dir, f"{task_id}.npy")
                np.save(mask_save_path, binary_mask)
                rel_mask = os.path.relpath(mask_save_path, PROJECT_ROOT).replace("\\", "/")

                desc_text = TASK06_EXPERT_DESC[organ_name]

                # ✅ Standardized Metadata
                meta_list.append({
                    "img_path": rel_img,
                    "mask_path": rel_mask,
                    "modality": modality_str,  # 🌟 Attribute
                    "description": desc_text,  # 🌟 Attribute
                    "organ": organ_name,  # 🌟 Attribute
                    "raw_organ_id": int(val),
                    "source": "MSD_Task06"
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

    print(f"🚀 Starting MSD Task06 (Lung) Preprocessing")
    print(
        f"✨ Strategy: {'Spatial 2.5D (Lung Window)' if USE_25D else '2D Multi-Window Pseudo-RGB'} | Workers: {NUM_WORKERS}")

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
            sid = fname.split(".")[0]
            if sid.startswith("._"): sid = sid[2:]

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