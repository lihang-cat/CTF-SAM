import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """标准的 Dice Loss"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        target = target.float()
        pred_flat = pred.flatten(1)
        target_flat = target.flatten(1)

        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class UniversalTextSegLoss(nn.Module):
    """
    
    -  BCE + Dice
    
    """
    def __init__(self):
        super().__init__()
        # 仅保留医学分割黄金组合
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = DiceLoss()

    def forward(self, pred_masks, gt_mask, has_object):
        device = pred_masks.device
        gt_mask = gt_mask.float()
        
        mask_flag = (has_object > 0).view(-1)
        num_positive = mask_flag.sum()

        loss_bce = torch.tensor(0.0, device=device, requires_grad=True)
        loss_dice = torch.tensor(0.0, device=device, requires_grad=True)

        if num_positive > 0:
            pred_pos = pred_masks[mask_flag]
            gt_pos = gt_mask[mask_flag]

            # 1. BCE Loss 
            loss_bce = self.bce_loss_fn(pred_pos, gt_pos)
            
            # 2. Dice Loss 
            loss_dice = self.dice_loss_fn(pred_pos, gt_pos)

        # 总损失BCE 和 Dice
        total_loss = loss_bce + loss_dice

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_bce": loss_bce.item(),
            "loss_dice": loss_dice.item(),
            "num_pos": num_positive.item()
        }