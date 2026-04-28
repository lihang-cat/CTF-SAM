import os
import torch
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tabulate import tabulate
from scipy.ndimage import distance_transform_edt, binary_erosion

# --- 导入你的自定义模块 ---
from models.clip_sam_net import ClipSamNet
from data.dataset import ClipSamDataset

# --- 尝试导入 EMA 库 ---
try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    HAS_EMA = False

# =============================================================================
# 配置区域
# ==============================================================================
EVAL_CONFIG = {
    # 路径设置
    'data_root': '/root/sj-tmp/Med-CLIP-SAM',
    'json_path': 'data/processed/total_test.json',
    'checkpoint_path': 'checkpoints/Med-CLIP-SAM-Model-v1/best_model.pth',

    # 结果保存根目录
    'save_vis_dir': 'vis_results_v1',

    # 硬件与模型参数
    'input_size': 256,
    'batch_size': 64,
    'num_workers': 16,

    'nsd_tolerance': 2.0, 

    # --- 可视化配置 ---
    'save_visuals': True,  # 是否保存常规的前 N 张图
    'vis_count': 5,        # 每个数据集保存前 5 张常规图

    'model_cfg': {
        'input_size': 256,
        'mask_size': 256,
    }
}

# ==============================================================================
# 医学大类映射函数 (对齐 MedSAM)
# ==============================================================================
def get_organ_category(source, organ):
    """根据 数据集 和 器官名称，精细化划分其所属的 4 大临床类别"""
    organ_lower = str(organ).lower()
    
    # 1. 肿瘤与病灶 (Tumors & Lesions)
    tumor_datasets = ["BraTS", "BUSI", "ISIC2018", "Kvasir-SEG", "TN3K", "SIIM_Pneumothorax", "MSD_Task06"]
    if source in tumor_datasets or (source == "KiTS19" and "tumor" in organ_lower):
        return "Tumors & Lesions"

    # 2. 正常解剖：心胸 (Normal: Cardiothoracic)
    cardio_datasets = ["ACDC", "CAMUS", "MSD_Task02", "ChestXray"]
    ts_cardio_organs = [
        "heart", "aorta", "lung", "pulmonary vein", "superior vena cava", "inferior vena cava", 
        "trachea", "rib", "sternum", "costal cartilage", "clavicula left", "clavicula right", 
        "atrial appendage left", "brachiocephalic trunk", "brachiocephalic vein left", 
        "brachiocephalic vein right", "subclavian artery left", "subclavian artery right"
    ]
    if source in cardio_datasets or (source == "TotalSegmentator" and organ_lower in ts_cardio_organs):
        return "Normal: Cardiothoracic"

    # 3. 正常解剖：腹盆腔 (Normal: Abdomen & Pelvis)
    abdomen_datasets = ["CHAOS_CT", "CHAOS_MRI", "MSD_Task05"]
    ts_abdomen_organs = [
        "liver", "spleen", "pancreas", "gallbladder", "stomach", "duodenum", "small bowel", "colon", 
        "kidney left", "kidney right", "left adrenal gland", "right adrenal gland", "bladder", "prostate", 
        "portal vein and splenic vein", "esophagus", "iliac artery left", "iliac artery right", 
        "iliac vena left", "iliac vena right"
    ]
    if source in abdomen_datasets or (source == "KiTS19" and organ_lower == "kidney") or (source == "TotalSegmentator" and organ_lower in ts_abdomen_organs):
        return "Normal: Abdomen & Pelvis"

    # 4. 正常解剖：脑、头颈与肌骨 (Normal: Brain, Neck & MSK)
    brain_datasets = ["MSD_Task04", "Nerve_US"]
    if source in brain_datasets or source == "TotalSegmentator":
        return "Normal: Brain, Neck & MSK"

    return "Uncategorized"


# ==============================================================================
# 工具函数：计算 Normalized Surface Distance (NSD)
# ==============================================================================
def compute_nsd(pred, gt, tolerance=2.0):
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return None  
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 0.0

    gt_border = gt ^ binary_erosion(gt)
    pred_border = pred ^ binary_erosion(pred)

    if np.sum(gt_border) == 0 or np.sum(pred_border) == 0:
        return 0.0

    gt_dist = distance_transform_edt(~gt_border)
    pred_dist = distance_transform_edt(~pred_border)

    gt_border_in_tol = np.sum((pred_dist <= tolerance) * gt_border)
    pred_border_in_tol = np.sum((gt_dist <= tolerance) * pred_border)

    nsd = (gt_border_in_tol + pred_border_in_tol) / (np.sum(gt_border) + np.sum(pred_border) + 1e-8)
    return float(nsd)

# ==============================================================================
# 核心评估函数 
# ==============================================================================
def evaluate(model, dataloader, device, config):
    model.eval()

    # 字典结构：metrics[dataset_name][organ_name]['dice']
    metrics = defaultdict(lambda: defaultdict(lambda: {'dice': [], 'nsd': []}))

    normal_counts = defaultdict(int)

    base_vis_dir = config['save_vis_dir']
    os.makedirs(base_vis_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    nsd_tol = config.get('nsd_tolerance', 2.0)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            imgs = batch['image_clip'].to(device)
            texts = batch['text_token'].to(device)
            masks = batch['gt_mask'].to(device)
            ds_names = batch['source']
            
            # 🌟 关键修复：分别获取短器官名（用于分类）和长句子（用于画图）
            organs = batch['organ']          
            prompts = batch['prompt_text']   

            with autocast(dtype=amp_dtype): 
                pred_logits, _, _ = model(imgs, texts)

            pred_masks = (pred_logits > 0.0).float()
            bs = imgs.shape[0]
            preds_flat = pred_masks.view(bs, -1)
            gts_flat = masks.view(bs, -1)

            intersection = (preds_flat * gts_flat).sum(dim=1)
            pred_sums = preds_flat.sum(dim=1)
            gt_sums = gts_flat.sum(dim=1)

            dice_tensor = (2. * intersection + 1e-5) / (pred_sums + gt_sums + 1e-5)
            dice_np = dice_tensor.cpu().numpy()
            pred_sums_np = pred_sums.cpu().numpy()
            gt_sums_np = gt_sums.cpu().numpy()

            for k in range(bs):
                dataset_name = ds_names[k]
                organ_name = organs[k]   # 🌟 使用纯净的 organ_name (如 'liver') 用于分类与制表
                full_prompt = prompts[k] # 🌟 使用长句子用于图片显示
                
                if gt_sums_np[k] == 0 and pred_sums_np[k] == 0:
                    continue
                
                pred_mask_k = pred_masks[k].squeeze().cpu().numpy().astype(bool)
                gt_mask_k = masks[k].squeeze().cpu().numpy().astype(bool)

                current_dice = float(dice_np[k])
                current_nsd = compute_nsd(pred_mask_k, gt_mask_k, tolerance=nsd_tol)

                # 按数据集和短器官名存入统计字典
                metrics[dataset_name][organ_name]['dice'].append(current_dice)
                if current_nsd is not None:
                    metrics[dataset_name][organ_name]['nsd'].append(current_nsd)

                # --- 常规可视化逻辑 ---
                if config['save_visuals'] and normal_counts[dataset_name] < config['vis_count']:
                    normal_dir = os.path.join(base_vis_dir, "normal_samples", dataset_name)
                    os.makedirs(normal_dir, exist_ok=True)
                    # 可视化依然传入 full_prompt，让图片上的字是完整的
                    save_visualization(imgs[k], masks[k], pred_masks[k], full_prompt, dataset_name, normal_dir, f"sample_{normal_counts[dataset_name]:03d}", current_dice, current_nsd)
                    normal_counts[dataset_name] += 1

    return metrics


def save_visualization(img_tensor, gt_tensor, pred_tensor, prompt, ds_name, save_dir, prefix, dice_score, nsd_score):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=img_tensor.device).view(3, 1, 1)

    img = (img_tensor * std + mean).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gt = gt_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255
    pred = pred_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255

    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    pred_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    h, w, _ = img_rgb.shape
    pad_h = 60  

    def format_panel(image, caption, d_score=None, n_score=None):
        canvas = cv2.copyMakeBorder(image, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        # 处理长文本溢出问题
        max_text_width = w - 20
        text_size = cv2.getTextSize(caption, font, 0.5, 1)[0]
        if text_size[0] > max_text_width:
            caption = caption[:30] + "..."
            text_size = cv2.getTextSize(caption, font, 0.5, 1)[0]
            
        cv2.putText(canvas, caption, ((w - text_size[0]) // 2, h + 25), font, 0.5, (0, 0, 0), 1)

        if d_score is not None and n_score is not None:
            score_str = f"DSC:{d_score*100:.1f}  NSD:{n_score*100:.1f}"
            score_font = cv2.FONT_HERSHEY_DUPLEX
            score_size = cv2.getTextSize(score_str, score_font, 0.5, 1)[0]
            cv2.putText(canvas, score_str, ((w - score_size[0]) // 2, h + 50), score_font, 0.5, (0, 0, 0), 1)
            
        return canvas

    concat_img = np.hstack((format_panel(img_rgb, f"(a) Image [{prompt}]"), format_panel(gt_rgb, "(b) Ground truth"), format_panel(pred_rgb, "(c) Prediction", d_score=dice_score, n_score=nsd_score)))
    cv2.imwrite(os.path.join(save_dir, f"{prefix}.jpg"), concat_img)


# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == '__main__':
    try:
        import scipy
    except ImportError:
        print("❌ 缺少 scipy 库。请运行: pip install scipy")
        exit()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    val_ds = ClipSamDataset(EVAL_CONFIG['json_path'], EVAL_CONFIG['data_root'], img_size=EVAL_CONFIG['input_size'], is_train=False)
    val_dl = DataLoader(val_ds, batch_size=EVAL_CONFIG['batch_size'], shuffle=False, num_workers=EVAL_CONFIG['num_workers'], pin_memory=True)

    print("🏗️  Building Model...")
    model = ClipSamNet(config=EVAL_CONFIG['model_cfg'], device=device).to(device)

    if os.path.exists(EVAL_CONFIG['checkpoint_path']):
        checkpoint = torch.load(EVAL_CONFIG['checkpoint_path'], map_location=device)
        if 'shadow_params' in checkpoint and HAS_EMA:
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
            ema.load_state_dict(checkpoint)
            ema.copy_to(model.parameters())
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)
        print("✅ Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found: {EVAL_CONFIG['checkpoint_path']}")

    print("🚀 Starting Evaluation...")
    raw_metrics = evaluate(model, val_dl, device, EVAL_CONFIG)

    # ==========================================================================
    # 🌟 统一数据对齐与制表 (百分制, Mean ± Std, 四大类汇总)
    # ==========================================================================
    all_results = []
    category_metrics = defaultdict(lambda: {'dice': [], 'nsd': []})
    table_data = []

    for ds_name, organs in sorted(raw_metrics.items()):
        for organ_name, scores in organs.items():
            cat_name = get_organ_category(ds_name, organ_name)
            
            dice_list = scores['dice']
            nsd_list = scores['nsd']
            
            # 百分制转换 + ddof=1 (无偏样本标准差)
            dice_mean = np.mean(dice_list) * 100 if dice_list else 0.0
            dice_std = np.std(dice_list, ddof=1) * 100 if len(dice_list) > 1 else 0.0
            nsd_mean = np.mean(nsd_list) * 100 if nsd_list else 0.0
            nsd_std = np.std(nsd_list, ddof=1) * 100 if len(nsd_list) > 1 else 0.0
            
            dice_str = f"{dice_mean:.1f} ± {dice_std:.1f}"
            nsd_str = f"{nsd_mean:.1f} ± {nsd_std:.1f}"
            
            all_results.append({
                "Category": cat_name,
                "Dataset": ds_name,
                "Organ": organ_name,
                "Count": len(dice_list),
                "Dice": dice_str,
                "NSD": nsd_str
            })
            
            # 用于最后算四大类宏平均
            if dice_list: category_metrics[cat_name]['dice'].append(dice_mean)
            if nsd_list:  category_metrics[cat_name]['nsd'].append(nsd_mean)

    # 构建终端展示表
    df = pd.DataFrame(all_results)
    df.sort_values(by=['Category', 'Dataset'], inplace=True)
    
    for _, row in df.iterrows():
        table_data.append([row['Category'][:15]+"...", row['Dataset'], row['Organ'][:15], row['Count'], row['Dice'], row['NSD']])

    table_data.append(["-" * 15, "-" * 15, "-" * 15, "-", "-" * 15, "-" * 15])
    
    # 追加四大类宏平均结果
    for cat_name, metrics in category_metrics.items():
        cat_dice_list = metrics['dice']
        cat_nsd_list = metrics['nsd']
        
        m_dice_mean = np.mean(cat_dice_list) if cat_dice_list else 0.0
        m_dice_std = np.std(cat_dice_list, ddof=1) if len(cat_dice_list) > 1 else 0.0
        m_nsd_mean = np.mean(cat_nsd_list) if cat_nsd_list else 0.0
        m_nsd_std = np.std(cat_nsd_list, ddof=1) if len(cat_nsd_list) > 1 else 0.0
        
        table_data.append([
            "【MACRO SUMMARY】", "---", cat_name[:15], len(cat_dice_list),
            f"{m_dice_mean:.1f} ± {m_dice_std:.1f}", 
            f"{m_nsd_mean:.1f} ± {m_nsd_std:.1f}"
        ])
        
        df.loc[len(df)] = {
            "Category": "【MACRO SUMMARY】", "Dataset": "---", "Organ": cat_name, 
            "Count": len(cat_dice_list), 
            "Dice": f"{m_dice_mean:.1f} ± {m_dice_std:.1f}", 
            "NSD": f"{m_nsd_mean:.1f} ± {m_nsd_std:.1f}"
        }

    print("\n" + "=" * 90)
    print("📊 ClipSamNet Clinical Evaluation Report (Aligned with MedSAM)")
    print("=" * 90)
    print(tabulate(table_data, headers=["Category", "Dataset", "Organ", "Count", "Dice (%)", "NSD (%)"]))

    # 保存一键可复制的 CSV 表格
    csv_path = os.path.join(EVAL_CONFIG['save_vis_dir'], "clipsamnet_clinical_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n🎉 评估完成！已生成包含四大类宏平均的顶刊格式表格: {csv_path}")