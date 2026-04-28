import os
import cv2
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 核心绘图函数
# ==============================================================================
def draw_overlay(image, gt_mask, pred_mask=None, gt_color=(255, 0, 0), pred_color=(0, 255, 0), thickness=2):
    """在原图上叠加 GT（红线）和 Pred（绿线）的精细轮廓"""
    overlay = image.copy().astype(np.uint8)
    
    if gt_mask is not None and gt_mask.max() > 0:
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
        gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, gt_contours, -1, gt_color, thickness)
        
    if pred_mask is not None and pred_mask.max() > 0:
        pred_mask = (pred_mask > 0).astype(np.uint8) * 255
        pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, pred_contours, -1, pred_color, thickness)
        
    return overlay

def generate_qualitative_matrix(samples, col_titles, save_base_name="Final_Paper_Figure_4_7"):
    """生成多模型对比 6 列矩阵大图，并同时导出 PDF 和 PNG"""
    num_rows = len(samples)
    num_cols = len(col_titles)
    
    # 设定 600 DPI 达到顶刊出版标准
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(3 * num_cols, 3 * num_rows), dpi=600)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    for row_idx, sample in enumerate(samples):
        img = sample['image']
        gt = sample['gt']
        
        # 严格按照你要求的 6 列顺序组装
        overlays = [
            draw_overlay(img, gt, None),                  # Col 0: Ground Truth
            draw_overlay(img, gt, sample.get('nnunet')),    # Col 1: nnU-Net
            draw_overlay(img, gt, sample.get('universeg')), # Col 2: UniverSeg
            draw_overlay(img, gt, sample.get('lvit')),      # Col 3: LViT
            draw_overlay(img, gt, sample.get('medsam')),    # Col 4: MedSAM (Tight)
            draw_overlay(img, gt, sample.get('ours'))       # Col 5: CTF-SAM (Ours)
        ]
        
        for col_idx, overlay in enumerate(overlays):
            ax = axes[row_idx, col_idx] if num_rows > 1 else axes[col_idx]
            ax.imshow(overlay)
            ax.axis('off')
            
            # 第一行添加列标题 (模型名称)
            if row_idx == 0:
                weight = 'bold' if col_idx == num_cols - 1 else 'normal'
                color = '#8B0000' if col_idx == num_cols - 1 else 'black'
                ax.set_title(col_titles[col_idx], fontsize=16, pad=12, weight=weight, color=color)
            
            # 第一列添加行标题 (数据集/器官名称)
            if col_idx == 0:
                ax.text(-0.1, 0.5, sample['name'], transform=ax.transAxes, 
                        fontsize=15, weight='bold', rotation=90, va='center', ha='right')

    # 底部图例
    fig.text(0.5, 0.08 if num_rows == 1 else 0.05, 
             '— Ground Truth (Red)      — Prediction (Green)', 
             ha='center', va='center', fontsize=15, weight='bold', color='black')

    # ==============================================================
    # 🌟 核心修改：同时保存 PDF 矢量图和 PNG 预览图
    # ==============================================================
    pdf_path = f"{save_base_name}.pdf"
    png_path = f"{save_base_name}.png"
    
    # 保存 PDF (论文 LaTeX 插入首选，无限放大不失真)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    
    # 保存 PNG (日常预览、发微信首选)
    plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.1)
    
    print(f"\n🎉 完美收官！论文级可视化大图已保存至:")
    print(f"   📄 矢量版 (推荐插入 LaTeX): {pdf_path}")
    print(f"   🖼️ 像素版 (方便本地预览): {png_path}")
    plt.close()

# ==============================================================================
# 2. 自动化评估与文件读取逻辑 (已补回强大的正则对齐匹配！)
# ==============================================================================
def compute_dice(pred, gt):
    """快速计算单张图的 Dice 分数"""
    pred_b = (pred > 0)
    gt_b = (gt > 0)
    if pred_b.sum() == 0 and gt_b.sum() == 0: return 1.0
    return 2.0 * np.logical_and(pred_b, gt_b).sum() / (pred_b.sum() + gt_b.sum() + 1e-8)

def auto_select_best_case(base_path, dataset):
    """遍历 CTF-SAM 文件夹，自动找出 Dice 分数最高的那个样本的绝对编号"""
    ctf_dir = os.path.join(base_path, dataset, "CTF-SAM")
    if not os.path.exists(ctf_dir):
        raise FileNotFoundError(f"❌ 找不到文件夹: {ctf_dir}")

    gt_files = glob.glob(os.path.join(ctf_dir, "*_gt.png"))
    best_sample_id = None
    best_dice = -1.0
    
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        
        # 🌟 核心：用正则表达式严格提取 "sample_0000" 这部分，无视后面的器官名！
        match = re.search(r'(sample_\d{4})', basename)
        if not match: continue
        sample_id = match.group(1)
        
        pred_files = glob.glob(os.path.join(ctf_dir, f"{sample_id}_*pred.png"))
        if not pred_files: continue
        pred_file = pred_files[0]
        
        gt = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        
        if gt is None or pred is None: continue
            
        dice = compute_dice(pred, gt)
        if dice > best_dice:
            best_dice = dice
            best_sample_id = sample_id
            
    if best_sample_id is None:
        raise ValueError(f"❌ 无法在 {dataset} 找到有效的预测掩码。")
        
    print(f"  ✨ [自动探测] 在 {dataset} 中锁定最佳样本: {best_sample_id} (CTF-SAM Dice: {best_dice*100:.2f}%)")
    return best_sample_id

def load_mask(base_path, dataset, model_dir, sample_id):
    """使用 sample_XXXX 进行无视后半截名称的模糊匹配读取"""
    search_pattern = os.path.join(base_path, dataset, model_dir, f"{sample_id}_*pred.png")
    files = glob.glob(search_pattern)
    if not files:
        print(f"    ⚠️ 警告: 找不到 {model_dir} 的预测结果，该图将留空。({search_pattern})")
        return None
    return cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)

def load_image_and_gt(base_path, dataset, sample_id):
    """从 CTF-SAM 文件夹加载绝对原图和真实标签"""
    img_pattern = os.path.join(base_path, dataset, "CTF-SAM", f"{sample_id}_*img.png")
    gt_pattern = os.path.join(base_path, dataset, "CTF-SAM", f"{sample_id}_*gt.png")
    
    img_files = glob.glob(img_pattern)
    gt_files = glob.glob(gt_pattern)
    
    if not img_files or not gt_files:
        raise FileNotFoundError(f"❌ 找不到 {sample_id} 的原图或 GT！")
        
    img = cv2.imread(img_files[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR 转为 RGB 给 plt 画图
    gt = cv2.imread(gt_files[0], cv2.IMREAD_GRAYSCALE)
    return img, gt

# ==============================================================================
# 3. 主程序配置区
# ==============================================================================
if __name__ == '__main__':
    # 绝对对齐你截图中的真实路径
    BASE_DIR = "/root/sj-tmp/Med-CLIP-SAM/vis_results_v1/export_for_paper"
    
    # 严格按照你截图中的 4 个数据集文件夹排布
    TARGET_DATASETS = [
        {
            "folder_name": "ChestXray",  
            "display_name": "Lung\n(Chest X-ray)"
        },
        {
            "folder_name": "CVC-ColonDB", 
            "display_name": "colon polyp\n(CVC-ColonDB)"
        },
        {
            "folder_name": "MSD_Task09", 
            "display_name": "spleen\n(MSD_Task09)"
        },
        {
            "folder_name": "PH2", 
            "display_name": "skin lesion\n(PH2)"
        }
    ]
    
    # 定义 6 列的矩阵标题
    COL_TITLES = ['Ground Truth', 'nnU-Net', 'UniverSeg', 'LViT', 'MedSAM (Tight)', 'CTF-SAM (Ours)']
    
    samples_data = []
    print("⏳ 开始执行全自动论文配图生成流水线...")
    
    for item in TARGET_DATASETS:
        ds = item["folder_name"]
        name = item["display_name"]
        
        print(f"\n👉 正在处理数据集: {ds}")
        
        # 1. 自动挑选该数据集中 CTF-SAM 表现最好（Dice最高）的图
        best_sample_id = auto_select_best_case(BASE_DIR, ds)
        
        # 2. 依据挑出的绝对编号，加载原图和 GT
        img, gt = load_image_and_gt(BASE_DIR, ds, best_sample_id)
        
        # 3. 顺藤摸瓜，去另外 4 个模型的文件夹里把这张图的预测结果抓过来
        mask_nnunet = load_mask(BASE_DIR, ds, "nnUNet", best_sample_id)
        mask_universeg = load_mask(BASE_DIR, ds, "UniverSeg", best_sample_id)
        mask_lvit = load_mask(BASE_DIR, ds, "LViT", best_sample_id)
        mask_medsam = load_mask(BASE_DIR, ds, "MedSAM_Tight", best_sample_id)
        mask_ours = load_mask(BASE_DIR, ds, "CTF-SAM", best_sample_id)
        
        # 4. 组装数据，送入绘图列队
        samples_data.append({
            'name': name,
            'image': img,
            'gt': gt,
            'nnunet': mask_nnunet,
            'universeg': mask_universeg,
            'lvit': mask_lvit,
            'medsam': mask_medsam,
            'ours': mask_ours
        })

    print("\n🎨 数据拼装完毕，正在绘制高分辨率 6 列矩阵大图...")
    # 保存基础文件名为 "Final_Paper_Figure_4_7"，函数内部会自动加 .pdf 和 .png 后缀
    generate_qualitative_matrix(samples_data, COL_TITLES, save_base_name="Final_Paper_Figure_4_7")