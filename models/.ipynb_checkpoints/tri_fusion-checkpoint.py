import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 基础组件：空间动态门控 (核心)
# ==============================================================================
class SpatialGatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 输入: 原始特征 + 新特征 (dim * 2)
        # 输出: 空间门控权重图 (1 channel)
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=1),
            nn.Sigmoid()  # 限制在 0~1
        )

    def forward(self, identity_feat, new_feat):
        """
        Args:
            identity_feat: 原始保留特征 (Identity) [B, C, H, W]
            new_feat: 新计算的特征 (Residual) [B, C, H, W]
        """
        # 1. 拼接
        concated = torch.cat([identity_feat, new_feat], dim=1)

        # 2. 计算权重 alpha (B, 1, H, W)
        alpha = self.gate_net(concated)

        # 3. 动态融合: (1-a) * old + a * new
        f_final = (1 - alpha) * identity_feat + alpha * new_feat

        return f_final


# ==============================================================================
# 基础组件：注意力块 (保持不变)
# ==============================================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=True, dropout=0.1):
        super().__init__()
        self.norm1_q = nn.LayerNorm(dim)
        self.norm1_k = nn.LayerNorm(dim)
        self.norm1_v = nn.LayerNorm(dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, bias=qkv_bias, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, query, key, value):
        q_norm, k_norm, v_norm = self.norm1_q(query), self.norm1_k(key), self.norm1_v(value)
        attn_out, _ = self.multihead_attn(query=q_norm, key=k_norm, value=v_norm)
        x = query + attn_out  # 注意: 这里的 Transformer 内部残差通常保持标准加法，以免破坏梯度流
        x = x + self.mlp(self.norm2(x))
        return x


# ==============================================================================
# 主模块：全门控三元融合 (All-Gated Fusion)
# ==============================================================================
class TriModalFusionModule(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        # 1. 空间先验投影
        self.heatmap_projector = nn.Sequential(
            nn.Conv2d(1, feature_dim // 4, kernel_size=1),
            nn.BatchNorm2d(feature_dim // 4),
            nn.GELU(),
            nn.Conv2d(feature_dim // 4, feature_dim, kernel_size=1)
        )

        # 2. 第一阶段：并行特征提取
        self.attn_img_text = CrossAttentionBlock(feature_dim, num_heads, dropout=dropout)
        self.attn_img_corr = CrossAttentionBlock(feature_dim, num_heads, dropout=dropout)
        self.attn_text_corr = CrossAttentionBlock(feature_dim, num_heads, dropout=dropout)

        # 3. 第二阶段：融合注意力
        self.fusion_attn = CrossAttentionBlock(feature_dim, num_heads, dropout=dropout)

        # -----------------------------------------------------------
        # [核心修改] 定义两个动态门控模块
        # -----------------------------------------------------------
        # Gate 1: 内部融合门控 (fusion_attn_out vs out_img_text)
        self.gate_internal = SpatialGatedResidual(feature_dim)

        # Gate 2: 最终输出门控 (fusion_block_out vs original_img_feat)
        self.gate_final = SpatialGatedResidual(feature_dim)

        # Final Norm
        self.norm_final = nn.LayerNorm(feature_dim)

    def forward(self, img_feat, text_feat, corr_map):
        """
        img_feat: [B, C, H, W]
        text_feat: [B, C]
        corr_map:[B, 1, H, W]
        """
        B, C, H, W = img_feat.shape
        L = H * W

        # --- Step 1: 准备 Flat 数据 ---
        f_corr = self.heatmap_projector(corr_map)

        img_flat = img_feat.flatten(2).transpose(1, 2)  # [B, L, C]
        corr_flat = f_corr.flatten(2).transpose(1, 2)  # [B, L, C]
        text_flat = text_feat.unsqueeze(1)  # [B, 1, C]

        # --- Step 2: 第一阶段 - 三路交互 ---
        # Q: Img, K,V: Text
        out_img_text = self.attn_img_text(query=img_flat, key=text_flat, value=text_flat)
        # Q: Img, K,V: Corr
        out_img_corr = self.attn_img_corr(query=img_flat, key=corr_flat, value=corr_flat)
        # Q: Text, K,V: Corr
        out_text_corr = self.attn_text_corr(query=text_flat, key=corr_flat, value=corr_flat)

        # --- Step 3: 第二阶段 - QKV 融合 ---
        fusion_q = out_img_text
        fusion_k = out_img_corr
        fusion_v = out_text_corr.expand(-1, L, -1)  # Broadcast text to image size

        # 计算融合注意力
        f_fused_flat = self.fusion_attn(query=fusion_q, key=fusion_k, value=fusion_v)

        # --- Step 4: [Gating 1] 内部动态残差 ---
        # 我们需要将 Flat 特征变回 Spatial 特征才能做 Conv 门控
        fusion_q_spatial = fusion_q.transpose(1, 2).reshape(B, C, H, W)
        f_fused_spatial_raw = f_fused_flat.transpose(1, 2).reshape(B, C, H, W)

        # Gate 1: 决定 "Attention 计算出的新关系" 有多少保留价值
        # 对比基准是 out_img_text (图文基础特征)
        f_stage1_spatial = self.gate_internal(fusion_q_spatial, f_fused_spatial_raw)

        # --- Step 5: [Gating 2] 最终动态残差 ---
        # Gate 2: 决定整个 "Fusion 模块" 对 "原始 CLIP 图像特征" 的增强程度
        # 对比基准是 img_feat (原始特征)
        f_final_spatial = self.gate_final(img_feat, f_stage1_spatial)

        # --- Step 6: 最终归一化 ---
        f_final = f_final_spatial.permute(0, 2, 3, 1)  # [B, H, W, C]
        f_final = self.norm_final(f_final)
        f_final = f_final.permute(0, 3, 1, 2)  # [B, C, H, W]

        return f_final