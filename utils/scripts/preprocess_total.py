import os
import glob
import random
import json
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from collections import Counter

# ================= 1. 配置区域 =================

DATASET_ROOT = r"D:\dataset\Totalsegmentator_dataset_v201"
OUTPUT_DIR = r"E:\code\Med-CLIP-SAM\data\processed\TotalSegmentator_256_Balanced_2.5D_Expert"
PROJECT_ROOT = r"E:\code\Med-CLIP-SAM"

IMG_SIZE = 256
MIN_PIXEL_THRESHOLD = 50  # 50 像素阈值
MIN_CATEGORY_COUNT_THRESHOLD = 500  # 🔴 新增：类别最小样本数阈值
SUBJECT_SAMPLE_RATE = 0.3  # 采样 30% 的病人
USE_25D = True  # 开启 2.5D
NUM_WORKERS = 8  # 自动设置核数

# ⭐⭐⭐ 核心优化：差异化采样配置 ⭐⭐⭐
# 格式: "类别关键词": (文件保留概率, 单文件最大切片数)
SAMPLING_CONFIG = {
    "rib": (0.3, 3),  # 肋骨：只处理 30% 的文件，每个文件只取 3 张
    "spine vertebra": (0.3, 3),
    "muscle": (0.5, 5),

    # === 大器官 (适度抑制/正常) ===
    "lung": (0.8, 3),  # 肺

    # === 关键软组织 (全力采集) ===
    "default": (1.0, 10)  # 100%保留，每个器官最多取 10 张
}


# ================= 2. 映射与描述 =================

class OrganMapper:
    def __init__(self):
        self.merge_rules = {
            "rib_": "rib",
            "vertebrae_": "spine vertebra",
            "costal_cartilages": "costal cartilage",
            "lung_": "lung",
            "kidney_cyst": "kidney cyst",
            "psoas_muscle": "muscle",
            "autochthon": "muscle"
        }
        self.rename_map = {
            "urinary_bladder": "bladder",
            "gallbladder": "gallbladder",
            "adrenal_gland_right": "right adrenal gland",
            "adrenal_gland_left": "left adrenal gland"
        }

    def get_organ_name(self, filename):
        name = filename.replace('.nii.gz', '').replace('.nii', '').lower()
        for k, v in self.merge_rules.items():
            if k in name: return v
        if name in self.rename_map: return self.rename_map[name]
        return name.replace("_", " ")


mapper = OrganMapper()


#
def get_description(organ_name):
    # 专家级解剖描述库

    desc_map = {
        # ================= 1. 核心实质性脏器 (Solid Organs) =================
        "liver": "liver, a large solid parenchymal organ in the right upper quadrant, presenting with uniform soft-tissue attenuation.",
        "spleen": "spleen, a solid highly vascular organ in the left upper quadrant, showing homogeneous soft-tissue density.",
        "kidney": "kidney, a paired bean-shaped retroperitoneal organ with distinct cortical and medullary soft-tissue enhancement.",
        "pancreas": "pancreas, a lobulated glandular organ lying transversely in the retroperitoneum posterior to the stomach.",
        "adrenal gland": "adrenal gland, a small, inverted Y-shaped endocrine gland located superior to the kidneys.",
        "prostate": "prostate, a small walnut-shaped glandular structure situated below the bladder in the male pelvis.",
        "uterus": "uterus, a pear-shaped muscular reproductive organ located in the female pelvis between the bladder and rectum.",

        # ================= 2. 空腔脏器 (Hollow Organs & GI Tract) =================
        "gallbladder": "gallbladder, a small fluid-filled sac located on the inferior surface of the liver, appearing hypodense.",
        "bladder": "urinary bladder, a hollow muscular organ in the anterior pelvis, typically presenting as a hypodense fluid-filled cavity.",
        "stomach": "stomach, a hollow muscular organ in the upper left abdomen, often containing a mixture of fluid, air, and soft-tissue density.",
        "esophagus": "esophagus, a collapsed tubular structure connecting the pharynx to the stomach, often containing trace amounts of air.",
        "small bowel": "small bowel, a tortuous tubular gastrointestinal structure with thin walls and varying internal contents.",
        "duodenum": "duodenum, the first C-shaped section of the small intestine adjacent to the pancreatic head.",
        "colon": "colon, the large intestine characterized by haustral markings, typically containing distinct air and fecal matter.",
        "rectum": "rectum, the terminal portion of the large intestine located in the posterior pelvis.",

        # ================= 3. 呼吸系统 (Respiratory System) =================
        "lung": "lung parenchyma, the primary respiratory organ in the thoracic cavity. In a soft-tissue window, it appears entirely black (hypodense) due to high air content.",
        "trachea": "trachea, the central tubular airway descending into the thorax, appearing strictly hypodense due to internal air.",

        # ================= 4. 心血管系统 (Cardiovascular System) =================
        "heart": "heart, a large muscular pump in the central mediastinum, encompassing multiple blood-filled chambers.",
        "aorta": "aorta, the main systemic artery appearing as a prominent tubular structure with fluid or contrast-enhanced density.",
        "inferior vena cava": "inferior vena cava, the largest systemic vein running parallel to the right of the aorta in the retroperitoneum.",
        "portal vein": "portal vein, a major vascular structure entering the liver hilum.",
        "pulmonary artery": "pulmonary artery, a large vascular structure originating from the right ventricle and branching into the lungs.",

        # ================= 5. 骨骼系统 (Skeletal System) =================
        "rib": "rib, a curved bony structure of the thoracic cage. In a soft-tissue window, it appears uniformly bright white (hyperdense).",
        "spine vertebra": "vertebra, a massive bony segment of the spinal column enclosing the spinal cord, presenting as hyperdense structures.",
        "femur": "femur, the large long bone of the thigh, articulating with the pelvis, showing highly hyperdense cortical bone.",
        "pelvis": "pelvic bone, the bony basin composed of the ilium, ischium, and pubis, appearing highly hyperdense.",
        "sacrum": "sacrum, a large wedge-shaped bone at the base of the spine, situated between the pelvic bones.",
        "clavicle": "clavicle, a horizontally oriented long bone in the superior thorax.",
        "scapula": "scapula, a flat triangular bone resting on the posterior thoracic wall.",
        "costal cartilage": "costal cartilage, segments of cartilage connecting the ribs to the sternum, often exhibiting distinct calcifications.",

        # ================= 6. 肌肉与脂肪 (Musculature & Adipose) =================
        "muscle": "skeletal muscle tissue, presenting with striated intermediate soft-tissue attenuation, delineating organ boundaries.",
        "psoas": "psoas major muscle, a thick, bilateral longitudinal skeletal muscle in the posterior abdominal wall.",
        "gluteus": "gluteal muscle, a large group of skeletal muscles forming the buttocks in the posterior pelvic region.",
        "autochthon": "autochthonous back muscles, the deep extensor muscles running longitudinally along the posterior spine."
    }


    # 模糊匹配逻辑
    for key, desc in desc_map.items():
        if key in organ_name:
            return desc

    # ⚠️ 关键修改：如果未命中，返回空字符串，而不是通用的废话
    return ""


# ================= 3. 核心处理函数 =================

def apply_window(img, center, width):
    lower = center - width / 2.0
    upper = center + width / 2.0
    img = np.clip(img, lower, upper)
    if upper == lower: return np.zeros_like(img)
    return (img - lower) / (upper - lower)


def get_25d_slice(vol_data, z_index):
    """ 2.5D with Soft Tissue Window """
    depth = vol_data.shape[2]
    W_CENTER, W_WIDTH = 40, 400
    idx_prev = max(0, z_index - 1)
    idx_curr = z_index
    idx_next = min(depth - 1, z_index + 1)

    s_prev = apply_window(vol_data[:, :, idx_prev], W_CENTER, W_WIDTH)
    s_curr = apply_window(vol_data[:, :, idx_curr], W_CENTER, W_WIDTH)
    s_next = apply_window(vol_data[:, :, idx_next], W_CENTER, W_WIDTH)

    return (np.stack([s_prev, s_curr, s_next], axis=-1) * 255).astype(np.uint8)


def resize_pad(image, target_size, is_mask=False):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left

    val = 0 if is_mask else [0, 0, 0]
    bord = cv2.BORDER_CONSTANT
    if is_mask:
        return cv2.copyMakeBorder(resized, top, bottom, left, right, bord, value=val)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, bord, value=val)


def get_sampling_params(organ_name):
    """根据器官名称获取采样参数"""
    # 优先匹配长关键词
    for key, params in SAMPLING_CONFIG.items():
        if key in organ_name:
            return params
    return SAMPLING_CONFIG["default"]


def process_single_subject(args):
    subject_path, output_img_dir, output_mask_dir = args
    meta_list = []

    subject_id = os.path.basename(subject_path)
    ct_path = os.path.join(subject_path, 'ct.nii.gz')
    if not os.path.exists(ct_path): return []

    try:
        ct_nii = nib.as_closest_canonical(nib.load(ct_path))
        ct_vol = ct_nii.get_fdata().astype(np.float32)

        seg_folder = os.path.join(subject_path, 'segmentations')
        mask_files = glob.glob(os.path.join(seg_folder, "*.nii.gz"))

        for mask_file in mask_files:
            mask_filename = os.path.basename(mask_file)
            organ_name = mapper.get_organ_name(mask_filename)

            # ⭐ 策略 1: 文件级 Dropout
            keep_prob, max_slices = get_sampling_params(organ_name)
            if random.random() > keep_prob:
                continue

            # Load Mask
            mask_nii = nib.as_closest_canonical(nib.load(mask_file))
            mask_vol = mask_nii.get_fdata().astype(np.uint8)
            if mask_vol.shape != ct_vol.shape: continue

            # Find valid slices
            z_sums = np.sum(mask_vol, axis=(0, 1))
            valid_slices = np.where(z_sums > 0)[0]
            if len(valid_slices) == 0: continue

            # ⭐ 策略 2: 切片级限制
            if len(valid_slices) > max_slices:
                selected_slices = sorted(random.sample(list(valid_slices), max_slices))
            else:
                selected_slices = valid_slices

            for z in selected_slices:
                mask_slice = mask_vol[:, :, z]

                # ⭐ 严格阈值过滤
                if np.sum(mask_slice) < MIN_PIXEL_THRESHOLD: continue

                # Processing
                if USE_25D:
                    img_processed = get_25d_slice(ct_vol, z)
                    modality_str = "CT "
                else:
                    norm = apply_window(ct_vol[:, :, z], 40, 400)
                    img_processed = np.stack([(norm * 255).astype(np.uint8)] * 3, axis=-1)
                    modality_str = "CT"

                img_padded = resize_pad(img_processed, IMG_SIZE, is_mask=False)
                mask_padded = resize_pad(mask_slice, IMG_SIZE, is_mask=True)

                # Save
                clean_raw_name = mask_filename.replace('.nii.gz', '')
                file_base_id = f"{subject_id}_{clean_raw_name}_{z}"

                img_path_save = os.path.join(output_img_dir, f"{file_base_id}.npy")
                mask_path_save = os.path.join(output_mask_dir, f"{file_base_id}.npy")

                np.save(img_path_save, img_padded)
                np.save(mask_path_save, mask_padded)

                # 🟢 获取描述：如果没有命中专家库，则为 ""
                description_text = get_description(organ_name)

                meta_list.append({
                    "img_path": os.path.relpath(img_path_save, PROJECT_ROOT).replace("\\", "/"),
                    "mask_path": os.path.relpath(mask_path_save, PROJECT_ROOT).replace("\\", "/"),
                    "modality": modality_str,
                    "description": description_text,  # 这里可能是空字符串
                    "organ": organ_name,
                    "raw_organ_id": clean_raw_name,
                    "source": "TotalSegmentator"
                })

    except Exception as e:
        print(f"Error {subject_id}: {e}")
        return []

    return meta_list


# ================= 4. Main =================

def main():
    img_dir = os.path.join(OUTPUT_DIR, "imgs")
    mask_dir = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"🚀 Starting Balanced TotalSegmentator Preprocessing")
    print(f"📉 Strategy: Dynamic Dropout + Strict Threshold > {MIN_PIXEL_THRESHOLD}px")

    all_subjects = sorted(glob.glob(os.path.join(DATASET_ROOT, "s*")))
    if not all_subjects: return

    # Sample Subjects
    sample_count = int(len(all_subjects) * SUBJECT_SAMPLE_RATE)
    sampled_subjects = random.sample(all_subjects, max(1, sample_count))
    print(f"✅ Processing {len(sampled_subjects)} subjects")

    tasks = [(s, img_dir, mask_dir) for s in sampled_subjects]
    all_metadata = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_subject, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(tasks)):
            res = future.result()
            if res: all_metadata.extend(res)

    # ==========================================================
    # 🔴 新增：严格过滤掉总数 < 500 的类别
    # ==========================================================
    print(f"\n🔍 Analyzing Class Counts (Threshold: {MIN_CATEGORY_COUNT_THRESHOLD})...")

    # 1. 统计当前所有样本的类别分布
    counts = Counter(item['organ'] for item in all_metadata)

    # 2. 区分合格类别与不合格类别
    valid_organs = {org for org, count in counts.items() if count >= MIN_CATEGORY_COUNT_THRESHOLD}
    dropped_organs = {org: count for org, count in counts.items() if count < MIN_CATEGORY_COUNT_THRESHOLD}

    # 3. 过滤 Metadata
    final_metadata = [item for item in all_metadata if item['organ'] in valid_organs]

    # 4. 打印过滤报告
    print("-" * 40)
    print(f"🚫 Dropped Categories (< {MIN_CATEGORY_COUNT_THRESHOLD} samples):")
    if dropped_organs:
        for org, count in sorted(dropped_organs.items(), key=lambda x: x[1]):
            print(f"   - {org}: {count}")
    else:
        print("   (None)")

    print("-" * 40)
    print(f"✅ Kept Categories (Top 15):")
    final_counts = Counter(item['organ'] for item in final_metadata)
    for org, count in final_counts.most_common(15):
        print(f"   - {org}: {count}")

    print(f"\n📉 Total samples before filter: {len(all_metadata)}")
    print(f"📈 Total samples after filter:  {len(final_metadata)}")

    # 5. 保存过滤后的 Metadata
    with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(final_metadata, f, indent=4)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()