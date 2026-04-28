
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import clip
# import math
#
# # 引入 LoRA 模块 (保持不变)
# try:
#     from models.LoRA import inject_lora_to_clip, mark_only_lora_as_trainable
# except ImportError:
#     def inject_lora_to_clip(*args, **kwargs):
#         pass
#
#
#     def mark_only_lora_as_trainable(*args, **kwargs):
#         pass
#
#
# # ==============================================================================
# # [新增] 自定义 Attention 类：显式分离 Q, K, V
# # ==============================================================================
# class LoRA_Compatible_Attention(nn.Module):
#     """
#     这是一个兼容 LoRA 的 Attention 替代品。
#     它将 nn.MultiheadAttention 的 in_proj_weight 拆解为独立的 q, k, v Linear 层。
#     这样 LoRA 就可以识别并注入 q_proj, k_proj, v_proj 了。
#     """
#
#     def __init__(self, original_mha):
#         super().__init__()
#         self.embed_dim = original_mha.embed_dim
#         self.num_heads = original_mha.num_heads
#         self.head_dim = self.embed_dim // self.num_heads
#         self.scale = self.head_dim ** -0.5
#
#         # 1. 创建独立的 Linear 层
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
#
#         # 复用原来的 out_proj (它本身就是 Linear，可以直接用)
#         self.out_proj = original_mha.out_proj
#
#         # 2. 从原始 MHA 中提取权重并赋值 (拆解 in_proj)
#         # in_proj_weight shape: [3*embed_dim, embed_dim]
#         # chunk(3) 顺序对应 q, k, v
#         q_w, k_w, v_w = original_mha.in_proj_weight.chunk(3, dim=0)
#         q_b, k_b, v_b = original_mha.in_proj_bias.chunk(3, dim=0)
#
#         with torch.no_grad():
#             self.q_proj.weight.copy_(q_w)
#             self.q_proj.bias.copy_(q_b)
#             self.k_proj.weight.copy_(k_w)
#             self.k_proj.bias.copy_(k_b)
#             self.v_proj.weight.copy_(v_w)
#             self.v_proj.bias.copy_(v_b)
#
#     def forward(self, query, key, value, attn_mask=None, need_weights=False):
#         # 输入 shape: [Seq_Len, Batch, Dim] (CLIP Visual 是 LND 格式)
#         L, N, E = query.shape
#
#         # 1. 独立计算 Q, K, V
#         q = self.q_proj(query)
#         k = self.k_proj(key)
#         v = self.v_proj(value)
#
#         # 2. Reshape 为 [Batch, Heads, Seq_Len, Head_Dim] 以进行 Attention
#         q = q.contiguous().view(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
#         k = k.contiguous().view(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
#         v = v.contiguous().view(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
#
#         # 3. Scaled Dot-Product Attention (使用 PyTorch 2.0 加速)
#         # output shape: [Batch, Heads, Seq_Len, Head_Dim]
#         output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
#
#         # 4. Reshape 回 [Seq_Len, Batch, Dim]
#         output = output.permute(2, 0, 1, 3).contiguous().view(L, N, E)
#
#         # 5. 输出投影
#         return self.out_proj(output), None
#
#
# class CLIPBackbone(nn.Module):
#     def __init__(self, model_name='ViT-B/16', device='cuda', use_lora=True, lora_r=4, lora_alpha=4):
#         super().__init__()
#         self.device = device if torch.cuda.is_available() else 'cpu'
#
#         print(f"Loading CLIP model: {model_name} on {self.device}...")
#         self.clip_model, _ = clip.load(model_name, device=self.device, jit=False)
#         self.clip_model = self.clip_model.float()
#
#         self.embed_dim = self.clip_model.visual.output_dim
#         if hasattr(self.clip_model.visual, 'width'):
#             self.visual_width = self.clip_model.visual.width
#         else:
#             self.visual_width = self.clip_model.visual.conv1.out_channels
#
#         # -----------------------------------------------------------
#         # LoRA 配置
#         # -----------------------------------------------------------
#         if use_lora:
#             print(f"Applying LoRA to CLIP Vision Backbone (Rank={lora_r}, Alpha={lora_alpha})...")
#
#             # [关键步骤 1] 先对模型进行“外科手术”，把合并的层拆开
#             self._split_qkv_for_lora()
#
#             # [关键步骤 2] 注入 LoRA (现在可以识别到 q_proj 等层了)
#             target_layers_list = [
#                 'c_fc', 'c_proj',  # MLP
#                 'out_proj',  # Attn Output
#                 'q_proj', 'k_proj', 'v_proj'  # [Fix] Attn Input (拆解后现在存在了)
#             ]
#
#             inject_lora_to_clip(
#                 self.clip_model.visual,
#                 r=lora_r,
#                 lora_alpha=lora_alpha,
#                 target_layers=target_layers_list
#             )
#             mark_only_lora_as_trainable(self.clip_model.visual)
#
#             # 冻结其他部分
#             self._freeze_module(self.clip_model.transformer)
#             self._freeze_module(self.clip_model.token_embedding)
#             self._freeze_module(self.clip_model.ln_final)
#             self.clip_model.positional_embedding.requires_grad = False
#             self.clip_model.visual.positional_embedding.requires_grad = False
#         else:
#             self._freeze_all_parameters()
#
#     def _split_qkv_for_lora(self):
#         """
#         [New] 遍历 Vision Transformer 的所有 Block，
#         将原生的 nn.MultiheadAttention 替换为支持 LoRA 的 LoRA_Compatible_Attention
#         """
#         print("🔧 Patching CLIP Attention blocks to expose Q/K/V for LoRA...")
#         vision_transformer = self.clip_model.visual.transformer
#
#         # 遍历每一层 ResBlock
#         for i, block in enumerate(vision_transformer.resblocks):
#             # 获取旧的 Attention 模块
#             old_attn = block.attn
#
#             # 创建新的模块 (自动继承权重)
#             new_attn = LoRA_Compatible_Attention(old_attn)
#
#             # 替换！
#             block.attn = new_attn.to(self.device)
#
#         print("✅ Patching complete. Q/K/V layers are now accessible.")
#
#     def _freeze_module(self, module):
#         for param in module.parameters():
#             param.requires_grad = False
#
#     def _freeze_all_parameters(self):
#         for param in self.clip_model.parameters():
#             param.requires_grad = False
#
#     def encode_text(self, text_tokens):
#         text_tokens = text_tokens.to(self.device)
#         x = self.clip_model.token_embedding(text_tokens)
#         x = x + self.clip_model.positional_embedding
#         x = x.permute(1, 0, 2)
#         x = self.clip_model.transformer(x)
#         x = x.permute(1, 0, 2)
#         x = self.clip_model.ln_final(x)
#         batch_indices = torch.arange(x.shape[0], device=self.device)
#         eos_indices = text_tokens.argmax(dim=-1)
#         x = x[batch_indices, eos_indices] @ self.clip_model.text_projection
#         return F.normalize(x, dim=-1)
#
#     def _resize_pos_embed(self, posemb, grid_size_h, grid_size_w):
#         cls_emb = posemb[0:1, :]
#         spatial_emb = posemb[1:, :]
#         orig_size = int(math.sqrt(spatial_emb.shape[0]))
#         if orig_size == grid_size_h and orig_size == grid_size_w:
#             return posemb
#         spatial_emb = spatial_emb.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
#         new_spatial_emb = F.interpolate(spatial_emb, size=(grid_size_h, grid_size_w), mode='bicubic',
#                                         align_corners=False)
#         new_spatial_emb = new_spatial_emb.permute(0, 2, 3, 1).reshape(-1, self.visual_width)
#         return torch.cat([cls_emb, new_spatial_emb], dim=0)
#
#     def encode_image_multiscale(self, image):
#         """
#         提取多层特征用于跳跃连接 (已修改以适配 MultiScaleDepthwiseAdapter)
#         Output:
#             final_spatial_512: [B, 512, 16, 16] (给 TriModalFusion)
#             global_feat: [B, 512]
#             features_list: List [[B, 768, 16, 16] * 7]
#                            对应 CLIP Layers: [Layer 11, 9, 7, 5, 3, 2, 1]
#         """
#         image = image.to(self.device)
#         vision_model = self.clip_model.visual
#         x = image.type(self.clip_model.dtype)
#
#         # Patch Embedding
#         x = vision_model.conv1(x)
#         B, C, H, W = x.shape
#         x = x.reshape(B, C, -1).permute(0, 2, 1)
#
#         # Pos Embedding
#         class_embedding = vision_model.class_embedding.to(x.dtype) + \
#                           torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=self.device)
#         x = torch.cat([class_embedding, x], dim=1)
#         resized_pos_embed = self._resize_pos_embed(vision_model.positional_embedding.to(x.dtype), H, W)
#         x = x + resized_pos_embed
#         x = vision_model.ln_pre(x)
#
#         # Transformer Loop
#         x = x.permute(1, 0, 2)  # LND
#
#         # [修改] 使用字典暂存所有需要的层，方便后续按特定顺序组装
#         features_dict = {}
#         # 我们需要的层索引 (0-indexed, CLIP ViT-B 共有 12 层, index 0~11)
#         # Layer 11, 9, 7, 5, 3, 2, 1
#         target_indices = [0, 1, 2, 4, 6, 8, 10]
#
#         for i, resblock in enumerate(vision_model.transformer.resblocks):
#             x = resblock(x)
#
#             if i in target_indices:
#                 # 提取特征
#                 feat = x.permute(1, 0, 2)  # [B, 257, 768]
#                 feat_spatial = feat[:, 1:, :]  # [B, 256, 768] (Remove class token)
#                 feat_spatial = feat_spatial.permute(0, 2, 1).reshape(B, self.visual_width, H, W)
#                 features_dict[i] = feat_spatial
#
#         x = x.permute(1, 0, 2)
#         x = vision_model.ln_post(x)
#         if vision_model.proj is not None:
#             x = x @ vision_model.proj
#
#         final_spatial_512 = x[:, 1:, :].permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
#         final_spatial_512 = F.normalize(final_spatial_512, dim=1)
#         global_feat = F.normalize(x[:, 0, :], dim=-1)
#
#         # [修改] 组装返回列表：必须严格按照 Adapter 要求的顺序 (从深到浅)
#         # Adapter 顺序: [11, 9, 7, 5, 3, 2, 1] -> 对应 index [10, 8, 6, 4, 2, 1, 0]
#         features_list = [
#             features_dict[10],  # Layer 11
#             features_dict[8],  # Layer 9
#             features_dict[6],  # Layer 7
#             features_dict[4],  # Layer 5
#             features_dict[2],  # Layer 3
#             features_dict[1],  # Layer 2
#             features_dict[0]  # Layer 1
#         ]
#
#         return final_spatial_512, global_feat, features_list
#
#     def forward(self, image, text_tokens):
#         img_spatial, txt_global, features_list = self.encode_image_multiscale(image)
#         txt_feat = self.encode_text(text_tokens)
#         return img_spatial, txt_feat, features_list

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