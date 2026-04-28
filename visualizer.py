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
    'json_path': 'data/processed/Task09_test.json', # Montgomery_ChestXray_test.json PH2_test.json Task09_test.json
    'checkpoint_path': 'checkpoints/CTF-SAM/best_model.pth',  

    # 结果保存根目录
    'save_vis_dir': 'vis_results_v5',

    # ==============================================================
    # 🌟 论文定性分析可视化导出配置
    # ==============================================================
    'export_masks': True,                                 
    'export_dir': 'vis_results_v1/export_for_paper',      
    'model_name': 'CTF-SAM',                              
    # ==============================================================

    # 硬件与模型参数
    'input_size': 256,
    'batch_size': 64,
    'num_workers': 16,

    'nsd_tolerance': 2.0, 

    # --- 可视化配置 ---
    'save_visuals': True,  
    'vis_count': 5,        

    'model_cfg': {
        'input_size': 256,
        'mask_size': 256,
    }
}


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

    global_idx = 0 

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            imgs = batch['image_clip'].to(device)
            texts = batch['text_token'].to(device)
            masks = batch['gt_mask'].to(device)
            ds_names = batch['source']
            
            organs = batch['organ']          
            prompts = batch['prompt_text']   
            img_paths = batch.get('img_path', [f"unknown" for _ in range(imgs.shape[0])]) 

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
                organ_name = organs[k]   
                full_prompt = prompts[k] 
                
                base_id = os.path.splitext(os.path.basename(img_paths[k]))[0]
                
                if gt_sums_np[k] == 0 and pred_sums_np[k] == 0:
                    global_idx += 1  
                    continue
                
                pred_mask_k = pred_masks[k].squeeze().cpu().numpy().astype(bool)
                gt_mask_k = masks[k].squeeze().cpu().numpy().astype(bool)

                current_dice = float(dice_np[k])
                current_nsd = compute_nsd(pred_mask_k, gt_mask_k, tolerance=nsd_tol)

                metrics[dataset_name][organ_name]['dice'].append(current_dice)
                if current_nsd is not None:
                    metrics[dataset_name][organ_name]['nsd'].append(current_nsd)

                if config['save_visuals'] and normal_counts[dataset_name] < config['vis_count']:
                    normal_dir = os.path.join(base_vis_dir, "normal_samples", dataset_name)
                    os.makedirs(normal_dir, exist_ok=True)
                    save_visualization(imgs[k], masks[k], pred_masks[k], full_prompt, dataset_name, normal_dir, f"sample_{normal_counts[dataset_name]:03d}", current_dice, current_nsd)
                    normal_counts[dataset_name] += 1

                # ==============================================================
                # 🌟 新增：导出对齐的预测掩码、GT 和 100%无损原图
                # ==============================================================
                if config.get('export_masks', True):
                    export_dir = os.path.join(config['export_dir'], dataset_name, config['model_name'])
                    os.makedirs(export_dir, exist_ok=True)
                    
                    sample_prefix = f"sample_{global_idx:04d}_{base_id}_{organ_name.replace(' ', '_')}"
                    
                    # 1. 导出预测掩码
                    pred_save = (pred_mask_k.astype(np.uint8) * 255)
                    cv2.imwrite(os.path.join(export_dir, f"{sample_prefix}_pred.png"), pred_save)
                    
                    # 2. 导出真实的 Ground Truth
                    gt_save = (gt_mask_k.astype(np.uint8) * 255)
                    cv2.imwrite(os.path.join(export_dir, f"{sample_prefix}_gt.png"), gt_save)
                    
                    # 3. 🌟 双保险原图导出：优先读取原始文件，失败则反归一化兜底
                    raw_img_saved = False
                    if img_paths[k] != "unknown":
                        full_img_path = os.path.join(config['data_root'], img_paths[k])
                        if os.path.exists(full_img_path):
                            if full_img_path.endswith('.npy'):
                                raw_img = np.load(full_img_path)
                                if len(raw_img.shape) == 3 and raw_img.shape[0] == 3:
                                    raw_img = raw_img.transpose(1, 2, 0)
                                if raw_img.max() <= 1.0:
                                    raw_img = (raw_img * 255)
                                raw_img = raw_img.astype(np.uint8)
                                if len(raw_img.shape) == 2 or raw_img.shape[2] == 1:
                                    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
                                else:
                                    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
                            else:
                                raw_img = cv2.imread(full_img_path)

                            # 确保尺寸对齐
                            if raw_img.shape[:2] != (256, 256):
                                raw_img = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                                
                            cv2.imwrite(os.path.join(export_dir, f"{sample_prefix}_img.png"), raw_img)
                            raw_img_saved = True

                    # 兜底：反归一化
                    if not raw_img_saved:
                        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(3, 1, 1)
                        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(3, 1, 1)
                        img_unnorm = (imgs[k] * std + mean).permute(1, 2, 0).cpu().numpy()
                        img_rgb = np.clip(img_unnorm * 255, 0, 255).astype(np.uint8)
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) 
                        cv2.imwrite(os.path.join(export_dir, f"{sample_prefix}_img.png"), img_bgr)
                # ==============================================================

                global_idx += 1  

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

    all_results = []
    table_data = []

    macro_dice_means = []
    macro_nsd_means = []

    for ds_name, organs in sorted(raw_metrics.items()):
        for organ_name, scores in sorted(organs.items()):
            dice_list = scores['dice']
            nsd_list = scores['nsd']
            
            dice_mean = np.mean(dice_list) * 100 if dice_list else 0.0
            dice_std = np.std(dice_list, ddof=1) * 100 if len(dice_list) > 1 else 0.0
            nsd_mean = np.mean(nsd_list) * 100 if nsd_list else 0.0
            nsd_std = np.std(nsd_list, ddof=1) * 100 if len(nsd_list) > 1 else 0.0
            
            dice_str = f"{dice_mean:.3f} ± {dice_std:.3f}"
            nsd_str = f"{nsd_mean:.3f} ± {nsd_std:.3f}"
            
            all_results.append({
                "Dataset": ds_name,
                "Organ": organ_name,
                "Count": len(dice_list),
                "Dice (%)": dice_str,
                "NSD (%)": nsd_str
            })
            
            if dice_list: macro_dice_means.append(dice_mean)
            if nsd_list:  macro_nsd_means.append(nsd_mean)

    df = pd.DataFrame(all_results)
    df.sort_values(by=['Dataset', 'Organ'], inplace=True)
    
    for _, row in df.iterrows():
        table_data.append([row['Dataset'], row['Organ'][:25], row['Count'], row['Dice (%)'], row['NSD (%)']])

    table_data.append(["-" * 15, "-" * 25, "-", "-" * 15, "-" * 15])
    
    m_dice_mean = np.mean(macro_dice_means) if macro_dice_means else 0.0
    m_dice_std = np.std(macro_dice_means, ddof=1) if len(macro_dice_means) > 1 else 0.0
    m_nsd_mean = np.mean(macro_nsd_means) if macro_nsd_means else 0.0
    m_nsd_std = np.std(macro_nsd_means, ddof=1) if len(macro_nsd_means) > 1 else 0.0
    
    global_macro_dice_str = f"{m_dice_mean:.3f} ± {m_dice_std:.3f}"
    global_macro_nsd_str = f"{m_nsd_mean:.3f} ± {m_nsd_std:.3f}"

    table_data.append([
        "【MACRO AVERAGE】", "All Datasets & Organs", len(macro_dice_means),
        global_macro_dice_str, 
        global_macro_nsd_str
    ])
    
    df.loc[len(df)] = {
        "Dataset": "【MACRO AVERAGE】", "Organ": "All Datasets & Organs", 
        "Count": len(macro_dice_means), 
        "Dice (%)": global_macro_dice_str, 
        "NSD (%)": global_macro_nsd_str
    }

    print("\n" + "=" * 90)
    print("📊 ClipSamNet Evaluation Report (Macro Average Over All Organs)")
    print("=" * 90)
    print(tabulate(table_data, headers=["Dataset", "Organ", "Count", "Dice (%)", "NSD (%)"]))

    csv_path = os.path.join(EVAL_CONFIG['save_vis_dir'], "clipsamnet_macro_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n🎉 评估完成！已生成包含全部器官宏平均的结果表格: {csv_path}")