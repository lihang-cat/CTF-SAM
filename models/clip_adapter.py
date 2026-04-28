
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
import math


class CLIPBackbone(nn.Module):
    def __init__(self, model_name='ViT-B/16', device='cuda'):
        super().__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'

        print(f"Loading CLIP model: {model_name} on {self.device}...")
        # jit=False 必须设置，否则无法求导
        self.clip_model, _ = clip.load(model_name, device=self.device, jit=False)

        # 将模型转为 float32，避免全参微调时混合精度的数值不稳定性
        self.clip_model = self.clip_model.float()

        self.embed_dim = self.clip_model.visual.output_dim
        if hasattr(self.clip_model.visual, 'width'):
            self.visual_width = self.clip_model.visual.width
        else:
            self.visual_width = self.clip_model.visual.conv1.out_channels

        # -----------------------------------------------------------
        # 冻结策略配置
        # -----------------------------------------------------------
        print("🔧 Configuring Fine-tuning Strategy:")

        # 1. 首先冻结所有参数 (包括 Text Encoder, logit_scale 等)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("  🔒 Text Encoder & Shared params: Frozen")

        # 2. 显式解冻 Visual Encoder 的所有参数 (全参微调)
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
        print("  🔓 Visual Encoder: Unfrozen (Full Parameter Tuning)")

    def encode_text(self, text_tokens):
        # 文本编码器处于冻结状态，但仍需计算梯度流的截断（虽然 requires_grad=False 自动处理了，加 no_grad 更显式）
        with torch.no_grad():
            text_tokens = text_tokens.to(self.device)
            x = self.clip_model.token_embedding(text_tokens)
            x = x + self.clip_model.positional_embedding
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x)
            batch_indices = torch.arange(x.shape[0], device=self.device)
            eos_indices = text_tokens.argmax(dim=-1)
            x = x[batch_indices, eos_indices] @ self.clip_model.text_projection
            return F.normalize(x, dim=-1)

    def _resize_pos_embed(self, posemb, grid_size_h, grid_size_w):
        cls_emb = posemb[0:1, :]
        spatial_emb = posemb[1:, :]
        orig_size = int(math.sqrt(spatial_emb.shape[0]))
        if orig_size == grid_size_h and orig_size == grid_size_w:
            return posemb
        spatial_emb = spatial_emb.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        new_spatial_emb = F.interpolate(spatial_emb, size=(grid_size_h, grid_size_w), mode='bicubic',
                                        align_corners=False)
        new_spatial_emb = new_spatial_emb.permute(0, 2, 3, 1).reshape(-1, self.visual_width)
        return torch.cat([cls_emb, new_spatial_emb], dim=0)

    def encode_image_multiscale(self, image):
        """
        提取多层特征 (适配 MultiScaleDepthwiseAdapter)
        Output:
            final_spatial_512: [B, 512, 16, 16]
            global_feat: [B, 512]
            features_list: List [[B, 768, 16, 16] * 7]
        """
        image = image.to(self.device)
        vision_model = self.clip_model.visual
        x = image.type(self.clip_model.dtype)

        # Patch Embedding
        x = vision_model.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)

        # Pos Embedding
        class_embedding = vision_model.class_embedding.to(x.dtype) + \
                          torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=self.device)
        x = torch.cat([class_embedding, x], dim=1)
        resized_pos_embed = self._resize_pos_embed(vision_model.positional_embedding.to(x.dtype), H, W)
        x = x + resized_pos_embed
        x = vision_model.ln_pre(x)

        # Transformer Loop
        x = x.permute(1, 0, 2)  # LND

        features_dict = {}
        # 对应 CLIP Layers: [Layer 11, 9, 7, 5, 3, 2, 1]
        target_indices = [0, 1, 2, 4, 6, 8, 10]

        for i, resblock in enumerate(vision_model.transformer.resblocks):
            x = resblock(x)

            if i in target_indices:
                # 提取特征
                feat = x.permute(1, 0, 2)
                feat_spatial = feat[:, 1:, :]  # 移除 class token
                feat_spatial = feat_spatial.permute(0, 2, 1).reshape(B, self.visual_width, H, W)
                features_dict[i] = feat_spatial

        x = x.permute(1, 0, 2)
        x = vision_model.ln_post(x)
        if vision_model.proj is not None:
            x = x @ vision_model.proj

        final_spatial_512 = x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
        final_spatial_512 = F.normalize(final_spatial_512, dim=1)
        global_feat = F.normalize(x[:, 0, :], dim=-1)

        # 组装返回列表
        features_list = [
            features_dict[10],  # Layer 11
            features_dict[8],  # Layer 9
            features_dict[6],  # Layer 7
            features_dict[4],  # Layer 5
            features_dict[2],  # Layer 3
            features_dict[1],  # Layer 2
            features_dict[0]  # Layer 1
        ]

        return final_spatial_512, global_feat, features_list

    def forward(self, image, text_tokens):
        img_spatial, txt_global, features_list = self.encode_image_multiscale(image)
        txt_feat = self.encode_text(text_tokens)
        return img_spatial, txt_feat, features_list