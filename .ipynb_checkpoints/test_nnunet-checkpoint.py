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
    'save_vis_dir': 'vis_results_v2',

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
    },
    
    
    'excluded_sources': ["TotalSegmentator"]
}

# ==============================================================================
# 医学大类映射函数 (8 大精细分类)
# ==============================================================================
def get_organ_category(source, organ):
    """8 大精细解剖与病理学分类 (Exclude TotalSegmentator)"""
    organ_lower = str(organ).lower()
    
    # ================= [病灶类] =================
    # 1. 实质脏器恶性肿瘤 (Solid Organ Tumors)
    solid_tumor_ds = ["BraTS", "BUSI", "MSD_Task06"]
    if source in solid_tumor_ds or (source == "KiTS19" and "tumor" in organ_lower) or (source == "LiTS" and "tumor" in organ_lower):
        return "T1: Solid Organ Tumors"
        
    # 2. 腔道与表面病损 (Hollow Organ & Surface Lesions)
    hollow_surface_ds = ["Kvasir-SEG", "CVC-ClinicDB", "ETIS-Larib", "ISIC2018"]
    if source in hollow_surface_ds:
        return "T2: Hollow & Surface Lesions"
        
    # 3. 结节与非典型异常 (Nodules & Atypical Anomalies)
    if source in ["TN3K", "SIIM_Pneumothorax"]:
        return "T3: Nodules & Anomalies"

    # ================= [正常器官类] =================
    # 4. 腹腔实质与消化器官 (Abdominal Solid & Digestive)
    if source == "CHAOS_CT" or (source == "CHAOS_MRI" and organ_lower in ["liver", "spleen"]) or (source == "LiTS" and organ_lower == "liver"):
        return "N1: Abdominal Solid Organs"
        
    # 5. 泌尿与生殖系统 (Genitourinary System)
    if source == "MSD_Task05" or (source == "CHAOS_MRI" and "kidney" in organ_lower) or (source == "KiTS19" and organ_lower == "kidney"):
        return "N2: Genitourinary System"
        
    # 6. 心血管系统 (Cardiovascular System)
    if source in ["ACDC", "CAMUS", "MSD_Task02"]:
        return "N3: Cardiovascular System"
        
    # 7. 呼吸系统 (Respiratory System)
    if source == "ChestXray":
        return "N4: Respiratory System"
        
    # 8. 神经与细微结构 (Nervous & Microstructures)
    if source in ["MSD_Task04", "Nerve_US", "DRIVE"]:
        return "N5: Nervous & Microstructures"

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

    metrics = defaultdict(lambda: defaultdict(lambda: {'dice': [], 'nsd': []}))
    normal_counts = defaultdict(int)

    base_vis_dir = config['save_vis_dir']
    os.makedirs(base_vis_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    nsd_tol = config.get('nsd_tolerance', 2.0)
    
    # 获取黑名单
    excluded_sources = config.get('excluded_sources', [])

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            imgs = batch['image_clip'].to(device)
            texts = batch['text_token'].to(device)
            masks = batch['gt_mask'].to(device)
            ds_names = batch['source']
            
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
                
                # 🌟 核心拦截逻辑：如果数据集在黑名单中，直接跳过当前样本
                if dataset_name in excluded_sources:
                    continue
                
                organ_name = organs[k]   
                full_prompt = prompts[k] 
                
                if gt_sums_np[k] == 0 and pred_sums_np[k] == 0:
                    continue
                
                pred_mask_k = pred_masks[k].squeeze().cpu().numpy().astype(bool)
                gt_mask_k = masks[k].squeeze().cpu().numpy().astype(bool)

                current_dice = float(dice_np[k])
                current_nsd = compute_nsd(pred_mask_k, gt_mask_k, tolerance=nsd_tol)

                # 🌟 修改：确保 dice 和 nsd 同步添加，防止列表长度不一致
                if current_nsd is not None:
                    metrics[dataset_name][organ_name]['dice'].append(current_dice)
                    metrics[dataset_name][organ_name]['nsd'].append(current_nsd)

                if config['save_visuals'] and normal_counts[dataset_name] < config['vis_count']:
                    normal_dir = os.path.join(base_vis_dir, "normal_samples", dataset_name)
                    os.makedirs(normal_dir, exist_ok=True)
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
    # 🌟 统一数据对齐与制表 (百分制, Mean ± Std, 8 大类汇总)
    # ==========================================================================
    all_results = []
    category_metrics = defaultdict(lambda: {'dice': [], 'nsd': []})
    table_data = []
    
    # 🌟 新增：用于保存给绘图脚本使用的原始分数记录 (Long Format)
    raw_plot_data = []

    for ds_name, organs in sorted(raw_metrics.items()):
        for organ_name, scores in organs.items():
            cat_name = get_organ_category(ds_name, organ_name)
            
            dice_list = scores['dice']
            nsd_list = scores['nsd']
            
            # 🌟 新增：展开记录当前类别所有的单次原始 Dice 和 NSD 分数
            for d_score, n_score in zip(dice_list, nsd_list):
                raw_plot_data.append({
                    "Region": cat_name,       
                    "Model": "CTF-SAM",       # 🌟 修改：模型名称更新为 CTF-SAM
                    "DSC": float(d_score),
                    "NSD": float(n_score)
                })
            
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
            
            if dice_list: category_metrics[cat_name]['dice'].append(dice_mean)
            if nsd_list:  category_metrics[cat_name]['nsd'].append(nsd_mean)

    df = pd.DataFrame(all_results)
    df.sort_values(by=['Category', 'Dataset'], inplace=True)
    
    # 🌟 优化截断长度，防止 8 大类名称显示不全
    for _, row in df.iterrows():
        table_data.append([row['Category'][:30], row['Dataset'], row['Organ'][:20], row['Count'], row['Dice'], row['NSD']])

    table_data.append(["-" * 25, "-" * 15, "-" * 15, "-", "-" * 15, "-" * 15])
    
    # 追加 8 大类宏平均结果 (按照 T1-T3, N1-N5 排序输出)
    for cat_name in sorted(category_metrics.keys()):
        metrics = category_metrics[cat_name]
        cat_dice_list = metrics['dice']
        cat_nsd_list = metrics['nsd']
        
        m_dice_mean = np.mean(cat_dice_list) if cat_dice_list else 0.0
        m_dice_std = np.std(cat_dice_list, ddof=1) if len(cat_dice_list) > 1 else 0.0
        m_nsd_mean = np.mean(cat_nsd_list) if cat_nsd_list else 0.0
        m_nsd_std = np.std(cat_nsd_list, ddof=1) if len(cat_nsd_list) > 1 else 0.0
        
        table_data.append([
            "【MACRO】 " + cat_name[:20], "---", "---", len(cat_dice_list),
            f"{m_dice_mean:.1f} ± {m_dice_std:.1f}", 
            f"{m_nsd_mean:.1f} ± {m_nsd_std:.1f}"
        ])
        
        df.loc[len(df)] = {
            "Category": "【MACRO SUMMARY】", "Dataset": "---", "Organ": cat_name, 
            "Count": len(cat_dice_list), 
            "Dice": f"{m_dice_mean:.1f} ± {m_dice_std:.1f}", 
            "NSD": f"{m_nsd_mean:.1f} ± {m_nsd_std:.1f}"
        }

    print("\n" + "=" * 100)
    print("📊 CTF-SAM Clinical Evaluation Report (8 Fine-grained Categories)")
    print("=" * 100)
    print(tabulate(table_data, headers=["Category", "Dataset", "Organ", "Count", "Dice (%)", "NSD (%)"]))

    # 保存一键可复制的 CSV 表格
    csv_path = os.path.join(EVAL_CONFIG['save_vis_dir'], "ctfsam_8_categories_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n🎉 评估完成！已生成剔除冗余数据的 8 大类精细评估表格: {csv_path}")

    # 🌟 新增：保存用于画箱线图的原始分布数据 (Long Format) CSV
    if raw_plot_data:
        df_plot = pd.DataFrame(raw_plot_data)
        plot_csv_path = os.path.join(EVAL_CONFIG['save_vis_dir'], "ctfsam_boxplot_raw_data.csv")
        df_plot.to_csv(plot_csv_path, index=False)
        print(f"📈 画图专用的原始分布数据已保存至: {plot_csv_path}")