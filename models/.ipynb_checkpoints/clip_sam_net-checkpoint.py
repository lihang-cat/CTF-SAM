import torch
import torch.nn as nn


# 导入我们可以之前写好的四个模块
from models.clip_adapter import CLIPBackbone
from models.prompt_generator import PromptGenerator
from models.tri_fusion import TriModalFusionModule
from models.sam_decoder import SamDecoderWrapper


class ClipSamNet(nn.Module):
    def __init__(self,
                 config,
                 device='cuda'):
        """
        TF-SAM: Tri-Modal Fusion SAM with Text-Guided Prompting

        Args:
            config: 包含超参数的配置对象或字典 (e.g., lora_rank, dim, etc.)
            device: 运行设备
        """
        super().__init__()
        self.config = config
        self.device = device

        # ---------------------------------------------------------------------
        # Stage 1: Feature Extraction (CLIP + LoRA)
        # ---------------------------------------------------------------------
        self.clip_backbone = CLIPBackbone(
            model_name='ViT-B/16',
            device=device,


        )

        # ---------------------------------------------------------------------
        # Stage 2: Prompt Generation (Correlation -> Box)
        # ---------------------------------------------------------------------
        self.prompt_generator = PromptGenerator(
            embed_dim=256,  # SAM embed dim
            input_image_size=(256, 256)  # 我们的统一工作分辨率
        )

        # ---------------------------------------------------------------------
        # Stage 3: Tri-Modal Deep Fusion (Paper Core)
        # ---------------------------------------------------------------------
        self.fusion_module = TriModalFusionModule(
            feature_dim=512,  # CLIP output dim
            num_heads=8,
            dropout=0.1
        )

        # ---------------------------------------------------------------------
        # Stage 4: Mask Decoding (SAM Decoder + Adapter)
        # ---------------------------------------------------------------------
        self.sam_decoder = SamDecoderWrapper(
            fusion_dim=512,  # Adapter Input (from Fusion)
            sam_embed_dim=256,  # Adapter Output (to Decoder)
            out_size=256,  # Final Mask Size
            fix_resol=64  # Decoder Internal Resolution
        )

        # 初始化状态
        self.to(device)

    def load_sam_checkpoint(self, checkpoint_path):
        """
        加载 SAM 官方预训练权重 (ViT-B)
        这个函数会自动把权重分发给 PromptGenerator 和 MaskDecoder
        """
        print(f"Loading SAM checkpoint from {checkpoint_path}...")
        try:
            state_dict = torch.load(checkpoint_path, map_location=self.device)

            # 1. 分发给 Prompt Encoder
            self.prompt_generator.load_pretrained_weights(state_dict)

            # 2. 分发给 Mask Decoder
            self.sam_decoder.load_pretrained_weights(state_dict)

            print("Successfully loaded SAM pretrained weights.")
        except FileNotFoundError:
            print(f"Error: SAM checkpoint not found at {checkpoint_path}!")
            exit(1)

    def forward(self, image, text_tokens):
        """
        Args:
            image: [B, 3, 256, 256] (CLIP Input)
            text_tokens: [B, 77] (CLIP Input)
        Returns:
            pred_masks: [B, 1, 256, 256] (Logits)
            corr_map:   [B, 1, 16, 16]   (For Aux Loss)
            boxes:      [B, 4]           (For Visualization/Debug)
        """

        # Step 1: CLIP Backbone (Vision + Language)
        # ---------------------------------------------------
        # img_feat: [B, 512, 16, 16]
        # text_feat: [B, 512]
        img_feat, text_feat,features_list = self.clip_backbone(image, text_tokens)

        # ---------------------------------------------------
        # Step 2: Weakly-Supervised Prompt Generation
        # ---------------------------------------------------
        # corr_map: [B, 1, 16, 16] -> 用于计算 AuxLoss
        # sparse:   [B, 1, 256]    -> Box Embeddings
        # dense:    [B, 256, 64, 64]-> No-Mask Embeddings
        sparse_emb, dense_emb, corr_map, boxes = self.prompt_generator(img_feat, text_feat)

        # ---------------------------------------------------
        # Step 3: Tri-Modal Semantic Fusion (Cross-Attention)
        # ---------------------------------------------------
        # fusion_feat: [B, 512, 16, 16]
        # 融合了 Image, Text, Heatmap 三者信息
        fusion_feat = self.fusion_module(img_feat, text_feat, corr_map)

        # ---------------------------------------------------
        # Step 4: Decoding & Segmentation
        # ---------------------------------------------------
        # pred_masks: [B, 1, 256, 256]
        # 注意：这里传入 dense_emb (其实是 no_mask_embed)
        pred_masks, _ = self.sam_decoder(fusion_feat,features_list,sparse_emb,dense_emb)

        return pred_masks, corr_map, boxes