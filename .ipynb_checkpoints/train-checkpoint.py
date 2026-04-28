import os
import logging
import math
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
# from torchinfo import summary
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    HAS_EMA = False
    print("⚠️ torch_ema not found. EMA disabled.")

try:
    import wandb
    HAS_WANDB = False
except ImportError:
    HAS_WANDB = False

# --- 自定义模块  ---
from models.clip_sam_net import ClipSamNet
from utils.loss import UniversalTextSegLoss
from data.dataset import ClipSamDataset


CONFIG = {
    'exp_name': 'CTF-SAM-personalized_train', 
    'data_root': '/root/sj-tmp/Med-CLIP-SAM',
    'json_path': 'data/processed/total_train.json',
    'val_json_path': 'data/processed/total_val.json',
    'sam_checkpoint': './checkpoints/sam_vit_b_01ec64.pth',
    'save_dir': './checkpoints',

    # --- 训练硬件策略 ---
    'batch_size': 42,
    'accum_iter': 3,  # 梯度累加步数
    'num_workers': 12,
    'prefetch_factor': 2,

    # --- 训练周期 ---
    'num_epochs': 40,
    'val_interval': 2,
    'unfreeze_epoch': 5,  
    
    # --- 学习率策略 ---
    'lr_base': 4e-4,
    'lr_backbone': 1e-5,
    'weight_decay': 0.1,
    'warmup_epochs': 5,

    # --- 模型参数 ---
    'input_size': 256,
    'mask_size': 256,

    # --- 高级技巧 ---
    'use_ema': True,
    'ema_decay': 0.999,
    'max_grad_norm': 1.0,
}


# ==============================================================================
# 工具函数
# ==============================================================================
def setup_env(seed=42):
    """环境设置与加速开关"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
    logging.basicConfig(
        filename=log_file, level=logging.INFO,
        format='[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m-%d %H:%M'))
    logging.getLogger('').addHandler(console)
    return logging.getLogger('')

def set_stage_status(model, epoch, unfreeze_epoch, logger):
    """阶段性解冻控制"""
    changed = False
    if epoch < unfreeze_epoch:
        for n, p in model.named_parameters():
            if 'clip_backbone' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True  
        if epoch == 0:
            logger.info("🔒 Stage 1: Freeze CLIP Backbone active.")
    else:
        if epoch == unfreeze_epoch:
            logger.info("🔓 Stage 2: Unfreezing CLIP Visual Only (Text stays Frozen)")
            changed = True 
            for n, p in model.named_parameters():
                p.requires_grad = True
    return changed

def build_optimizer(model, cfg, steps_per_epoch):
    """构建优化器与 Scheduler"""
    backbone_params = []
    head_params = []
    no_decay = {'bias', 'layer_norm.weight', 'ln.weight', 'bn.weight'}

    for n, p in model.named_parameters():
        is_backbone = 'clip_backbone' in n
        is_no_decay = any(nd in n for nd in no_decay)
        
       
        param_dict = {'params': [p]}
        param_dict['weight_decay'] = 0.0 if is_no_decay else cfg['weight_decay']

        if is_backbone:
            param_dict['lr'] = cfg['lr_backbone']
            backbone_params.append(param_dict)
        else:
            param_dict['lr'] = cfg['lr_base']
            head_params.append(param_dict)

    optimizer = optim.AdamW(backbone_params + head_params, weight_decay=cfg['weight_decay'])

    def lr_lambda(current_step):
        warmup_steps = cfg['warmup_epochs'] * steps_per_epoch
        total_steps = cfg['num_epochs'] * steps_per_epoch

        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
       
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def validate(model, dataloader, device, epoch, ema=None, logger=None):
    """验证函数：自动应用 EMA 权重进行评估"""
    if ema:
        ema.store()  
        ema.copy_to()  

    model.eval()
    dice_scores = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            imgs = batch['image_clip'].to(device)
            texts = batch['text_token'].to(device)
            masks = batch['gt_mask'].to(device)

            with autocast(dtype=amp_dtype):
                pred_masks, _, _ = model(imgs, texts)

            pred_bin = (pred_masks > 0.0).float()
            
            intersection = (pred_bin * masks).sum(dim=(1, 2, 3))
            union = pred_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            dice = (2. * intersection + 1e-5) / (union + 1e-5)
            dice_scores.extend(dice.cpu().numpy())
            
    if ema: ema.restore()

    mean_dice = np.mean(dice_scores)
    if HAS_WANDB and wandb.run is not None:
        wandb.log({"val/dice": mean_dice, "epoch": epoch})

    return mean_dice


# ==============================================================================
# 主程序
# ==============================================================================
def main():
    setup_env(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = os.path.join(CONFIG['save_dir'], CONFIG['exp_name'])
    logger = setup_logger(exp_dir)
    logger.info(f"Config: {CONFIG}")

    if HAS_WANDB:
        wandb.init(project="Med-CLIP-SAM", name=CONFIG['exp_name'], config=CONFIG)

    if torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
        logger.info("🚀 Using BFloat16 (BF16) for training.")
    else:
        amp_dtype = torch.float16
        logger.info("⚠️ BF16 not supported. Using Float16.")

    train_ds = ClipSamDataset(CONFIG['json_path'], CONFIG['data_root'], img_size=CONFIG['input_size'], is_train=True)
    val_ds = ClipSamDataset(CONFIG['val_json_path'], CONFIG['data_root'], img_size=CONFIG['input_size'], is_train=False)

    train_dl = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=True,
        drop_last=True, persistent_workers=True,
        prefetch_factor=CONFIG['prefetch_factor']
    )
    val_dl = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True
    )

    model = ClipSamNet(config=CONFIG, device=device).to(device)
    



    if os.path.exists(CONFIG['sam_checkpoint']):
        model.load_sam_checkpoint(CONFIG['sam_checkpoint'])
        logger.info(f"Loaded SAM checkpoint: {CONFIG['sam_checkpoint']}")

    
    effective_steps_per_epoch = math.ceil(len(train_dl) / CONFIG['accum_iter'])

    set_stage_status(model, 0, CONFIG['unfreeze_epoch'], logger)
    optimizer, scheduler = build_optimizer(model, CONFIG, effective_steps_per_epoch)

    criterion = UniversalTextSegLoss().to(device)
    scaler = GradScaler()
    ema = ExponentialMovingAverage(model.parameters(), decay=CONFIG['ema_decay']) if (
            CONFIG['use_ema'] and HAS_EMA) else None

    if ema: logger.info(f"✅ EMA enabled with decay {CONFIG['ema_decay']}")

    best_dice = 0.0
    global_step = 0

    logger.info("🚀 Start Training...")

    for epoch in range(CONFIG['num_epochs']):
        if set_stage_status(model, epoch, CONFIG['unfreeze_epoch'], logger):
            logger.info("🔓 Stage Change detected: CLIP Backbone successfully unfrozen.")

        model.train()
        epoch_loss = 0.0
        
        
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Ep {epoch + 1}/{CONFIG['num_epochs']}")

        for step, batch in enumerate(pbar):
            global_step += 1
            imgs = batch['image_clip'].to(device, non_blocking=True)
            texts = batch['text_token'].to(device, non_blocking=True)
            masks = batch['gt_mask'].to(device, non_blocking=True)
            has_obj = batch['has_object'].to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                
                pred_masks, _, _ = model(imgs, texts)
                
                
                loss, loss_dict = criterion(pred_masks, masks, has_obj)
                loss = loss / CONFIG['accum_iter']

            if not torch.isfinite(loss):
                logger.warning(f"⚠️ Loss NaN at Ep {epoch} Step {step}. Resetting Scaler.")
                scaler = GradScaler()
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            
            if (step + 1) % CONFIG['accum_iter'] == 0 or (step + 1) == len(train_dl):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

               
                if scale_before <= scaler.get_scale():
                    scheduler.step()

                if ema: ema.update()

            loss_val = loss.item() * CONFIG['accum_iter']
            epoch_loss += loss_val

            
            pbar.set_postfix({
                'Loss': f"{loss_val:.4f}",
                'Dice': f"{1.0 - loss_dict.get('loss_dice', 0):.3f}", 
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            if HAS_WANDB and wandb.run is not None and global_step % 50 == 0:
                wandb.log({
                    "train/loss": loss_val,
                    "train/lr": optimizer.param_groups[0]['lr']
                })

        avg_loss = epoch_loss / len(train_dl)
        logger.info(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        # --- Validation ---
        if (epoch + 1) % CONFIG['val_interval'] == 0 or (epoch + 1) == CONFIG['num_epochs']:
            val_dice = validate(model, val_dl, device, epoch, ema, logger)
            logger.info(f"📊 Val | Dice: {val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice

                if ema:
                    logger.info("💾 Saving EMA weights to best_model.pth ...")
                    ema.store()  
                    ema.copy_to()  

                torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))

                if ema:
                    torch.save(ema.state_dict(), os.path.join(exp_dir, "best_ema.pth"))
                    ema.restore()

                logger.info(f"🏆 Best Model Saved (Dice: {best_dice:.4f})")

            torch.save(model.state_dict(), os.path.join(exp_dir, "latest_model.pth"))
            if ema: torch.save(ema.state_dict(), os.path.join(exp_dir, "latest_ema.pth"))

    logger.info(f"Finish. Best Dice: {best_dice:.4f}")
    if HAS_WANDB and wandb.run is not None: wandb.finish()


if __name__ == '__main__':
    main()