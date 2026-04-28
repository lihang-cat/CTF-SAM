
from segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.modeling.prompt_encoder import PositionEmbeddingRandom



import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type
from segment_anything.modeling import MaskDecoder


class TrainableMaskDecoder(MaskDecoder):
    """
    针对批量训练优化的 MaskDecoder  。
    修复了官方代码中 image_embeddings 强制 repeat_interleave 导致的维度爆炸问题。
    """

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测掩码。
        增加了“智能广播”逻辑：如果图像 Batch 已经匹配 Prompt Batch，则跳过重复操作。
        """

        # 1. 拼接 Output Tokens (IoU Token + Mask Tokens)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        # 将提示词 Embeddings 拼接
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # ------------------------------------------------------------------
        # 【修改点 1】: 图像特征 (src) 智能广播
        # 如果 image_embeddings 的 Batch (B) 与 tokens 的 Batch 一致，不进行重复
        # 否则 (例如单图多提示推理) 进行重复以匹配维度
        # ------------------------------------------------------------------
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        # 此时 src 和 dense_prompt_embeddings 维度均为 [B, C, H, W]
        src = src + dense_prompt_embeddings

        # ------------------------------------------------------------------
        # 【修改点 2】: 位置编码 (pos_src) 智能广播
        # 同理，如果 image_pe 已经具有 B 维度且匹配，则不重复
        # ------------------------------------------------------------------
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe

        # 获取当前维度
        b, c, h, w = src.shape

        # 2. 进入 Transformer 进行图文交互
        # hs: [B, N_tokens, C], src: [B, C, H, W]
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # 3. 上采样并预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        # 通过矩阵乘法生成掩码
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 4. 生成 IoU 预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred




class ConvBlock(nn.Module):
    """

    结构: Conv3x3 ->GroupNorm -> GELU -> Conv3x3 -> GroupNorm
    """

    def __init__(self, dim):
        super().__init__()


        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, dim)
        self.act = nn.GELU()


        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, dim)

        # 3. 动态残差缩放
        self.residual_scale = nn.Parameter(torch.ones(1) * 1e-2)

    def forward(self, x):
        identity = x

        # Block 1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # Block 2
        x = self.conv2(x)
        x = self.norm2(x)

        # Residual Connection
        return identity + self.residual_scale * x


class MultiScaleDepthwiseAdapter(nn.Module):
    def __init__(self, clip_dim=768, fusion_dim=512, out_dim=256):
        """

        Resolution Path: 16 -> 24 -> 32 -> 40 -> 48 -> 56 -> 64
        Skip Connections: ViT Layers [11, 9, 7, 5, 3, 2, 1]
        """
        super().__init__()

        # --- 1. 特征投影 (Feature Projectors) ---
        # 针对 TriModalFusion 的输出
        self.proj_fusion = nn.Sequential(
            nn.Conv2d(fusion_dim, out_dim, 1, bias=False),
            nn.GroupNorm(32, out_dim), nn.GELU()
        )

        # 针对 7 个 CLIP 跳跃连接层的投影 (共享结构，独立参数)
        # Layer 11, 9, 7, 5, 3, 2, 1
        self.skips_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(clip_dim, out_dim, 1, bias=False),
                nn.GroupNorm(32, out_dim), nn.GELU()
            ) for _ in range(7)
        ])

        # --- 2. 渐进式上采样模块 (Progressive Stages) ---
        # 我们定义 6 个上采样阶段，每个阶段负责提升分辨率并融合特征
        # 目标分辨率: 16 -> 24 -> 32 -> 40 -> 48 -> 56 -> 64

        # Stage 0: Base 16x16 (Fusion + Layer 11)
        self.stage0_block = ConvBlock(out_dim)

        # Stage 1: 16 -> 24 (Fusion + Layer 9)
        self.stage1_block = ConvBlock(out_dim)

        # Stage 2: 24 -> 32 (Fusion + Layer 7)
        self.stage2_block = ConvBlock(out_dim)

        # Stage 3: 32 -> 40 (Fusion + Layer 5)
        self.stage3_block = ConvBlock(out_dim)

        # Stage 4: 40 -> 48 (Fusion + Layer 3)
        self.stage4_block = ConvBlock(out_dim)

        # Stage 5: 48 -> 56 (Fusion + Layer 2)
        self.stage5_block = ConvBlock(out_dim)

        # Stage 6: 56 -> 64 (Fusion + Layer 1)
        self.stage6_block = nn.Sequential(
            ConvBlock(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),  # 平滑层
            nn.GroupNorm(32, out_dim),
            nn.GELU()
        )

        # 最终投影
        self.final_proj = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, fusion_feat, features_list):
        """
        Args:
            fusion_feat: [B, 512, 16, 16]
            features_list: List [[B, 768, 16, 16] * 7]
                           顺序对应 CLIP Layers: [11, 9, 7, 5, 3, 2, 1]
                           (需要在 CLIPBackbone 中修改输出列表顺序)
        """
        # 1. 基础特征对齐
        x = self.proj_fusion(fusion_feat)  # [B, 256, 16, 16]

        # 2. 提取并投影所有跳跃连接特征
        # list index 0->L11, 1->L9, 2->L7, 3->L5, 4->L3, 5->L2, 6->L1
        skips = [proj(feat) for proj, feat in zip(self.skips_projectors, features_list)]

        # --- Stage 0: 16x16 (Base Fusion) ---
        # 融合 Layer 11 (最深层语义)
        x = x + skips[0]
        x = self.stage0_block(x)

        # --- Stage 1: 16 -> 24 ---
        # 融合 Layer 9
        x = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)
        feat_skip = F.interpolate(skips[1], size=(24, 24), mode='bilinear', align_corners=False)
        x = x + feat_skip
        x = self.stage1_block(x)

        # --- Stage 2: 24 -> 32 ---
        # 融合 Layer 7
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
        feat_skip = F.interpolate(skips[2], size=(32, 32), mode='bilinear', align_corners=False)
        x = x + feat_skip
        x = self.stage2_block(x)

        # --- Stage 3: 32 -> 40 ---
        # 融合 Layer 5
        x = F.interpolate(x, size=(40, 40), mode='bilinear', align_corners=False)
        feat_skip = F.interpolate(skips[3], size=(40, 40), mode='bilinear', align_corners=False)
        x = x + feat_skip
        x = self.stage3_block(x)

        # --- Stage 4: 40 -> 48 ---
        # 融合 Layer 3
        x = F.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        feat_skip = F.interpolate(skips[4], size=(48, 48), mode='bilinear', align_corners=False)
        x = x + feat_skip
        x = self.stage4_block(x)

        # --- Stage 5: 48 -> 56 ---
        # 融合 Layer 2
        x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
        feat_skip = F.interpolate(skips[5], size=(56, 56), mode='bilinear', align_corners=False)
        x = x + feat_skip
        x = self.stage5_block(x)

        # --- Stage 6: 56 -> 64 ---
        # 融合 Layer 1 (最浅层细节)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        feat_skip = F.interpolate(skips[6], size=(64, 64), mode='bilinear', align_corners=False)
        x = x + feat_skip
        x = self.stage6_block(x)

        return self.final_proj(x)




# ======================================================================
# 3. SamDecoderWrapper
# ======================================================================
class SamDecoderWrapper(nn.Module):
    def __init__(
            self,
            fusion_dim: int = 512,
            sam_embed_dim: int = 256,
            fix_resol: int = 64,
            out_size: int = 256,
    ):
        super().__init__()
        self.fix_resol = fix_resol
        self.out_size = out_size

        self.adapter = MultiScaleDepthwiseAdapter(
            fusion_dim=fusion_dim,
            out_dim=out_size,
        )

        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=sam_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        )


        self.decoder = TrainableMaskDecoder(
            transformer_dim=sam_embed_dim,
            transformer=transformer,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.pe_layer = PositionEmbeddingRandom(sam_embed_dim // 2)

    def load_pretrained_weights(self, sam_state_dict: dict):
        decoder_dict = {}
        for k, v in sam_state_dict.items():
            if k.startswith("mask_decoder."):
                decoder_dict[k[len("mask_decoder."):]] = v

        # TrainableMaskDecoder 结构与原版完全一致，权重通用
        msg = self.decoder.load_state_dict(decoder_dict, strict=False)
        print(f"[SAM Decoder] Pretrained weights loaded: {msg}")

    def forward(
            self,
            fusion_feat: torch.Tensor,
            features_list: List[torch.Tensor],
            sparse_embeddings: torch.Tensor,
            dense_embeddings: torch.Tensor | None = None,
    ):
        B = fusion_feat.shape[0]

        image_embeddings = self.adapter(fusion_feat,features_list)

        image_pe = self.pe_layer(
            (self.fix_resol, self.fix_resol)
        ).unsqueeze(0).repeat(B, 1, 1, 1)

        low_res_masks, iou_predictions = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        masks = F.interpolate(
            low_res_masks,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )

        return masks, iou_predictions