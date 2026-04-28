import torch
import torch.nn as nn
import numpy as np
import cv2
from segment_anything.modeling import PromptEncoder

class PromptGenerator(nn.Module):
    def __init__(self,
                 embed_dim=256,
                 image_embedding_size=(64, 64),
                 input_image_size=(256, 256),
                 mask_in_chans=16,
                 activation=nn.GELU
                 ):
        super().__init__()
        self.input_image_size = input_image_size
        self.embed_dim = embed_dim

        # 1. SAM Prompt Encoder
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=input_image_size,
            mask_in_chans=mask_in_chans,
            activation=activation
        )

        

    def load_pretrained_weights(self, state_dict):
        my_dict = {}
        prefix = "prompt_encoder."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                name = k[len(prefix):]
                my_dict[name] = v
        if len(my_dict) > 0:
            msg = self.sam_prompt_encoder.load_state_dict(my_dict, strict=True)
            print(f"Loaded SAM Prompt Encoder weights: {msg}")

    def compute_correlation_map(self, img_feat, text_feat):
        sim_map = torch.einsum('bchw,bc->bhw', img_feat, text_feat)
        sim_map = sim_map.unsqueeze(1)  # [B, 1, H, W]
        return sim_map

    @torch.no_grad() 
    def generate_bounding_boxes_otsu(self, corr_map):
        """
        利用 Otsu 大津法从 CLIP 热力图中提取边界框 
        流水线：高斯平滑 -> Otsu二值化 -> 形态学闭运算 -> 最大连通域保留
        """
        B = corr_map.shape[0]
        boxes = []
        
        # 将热力图放大到原图尺寸 (256x256)
        corr_map_up = torch.nn.functional.interpolate(
            corr_map, size=self.input_image_size, mode='bilinear', align_corners=False
        )

        for i in range(B):
            cmap = corr_map_up[i, 0].detach().cpu().numpy()
            
            # 1. 归一化到 0-255
            cmap_min, cmap_max = cmap.min(), cmap.max()
            if cmap_max > cmap_min:
                cmap_norm = (cmap - cmap_min) / (cmap_max - cmap_min)
            else:
                cmap_norm = cmap
            cmap_8u = (cmap_norm * 255).astype(np.uint8)
            
            # 🔥 优化 1：高斯平滑，消除高频突刺噪点
            cmap_blurred = cv2.GaussianBlur(cmap_8u, (5, 5), 0)
            
            # 2. 应用大津法 (Otsu) 寻找最佳阈值并二值化
            _, binary_map = cv2.threshold(cmap_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 🔥 优化 2：形态学闭运算 (填补内部空洞，平滑边缘)
            kernel = np.ones((5, 5), np.uint8)
            binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
            
            # 3. 连通域分析，去除离散噪点，只保留最大连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
            
            if num_labels > 1:
                # 找到面积最大的前景连通域 (跳过索引 0 的背景)
                # cv2.CC_STAT_AREA 是面积的索引
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                
                # 仅保留最大连通域的像素
                y_indices, x_indices = np.where(labels == largest_label)
                
                x1, x2 = np.min(x_indices), np.max(x_indices)
                y1, y2 = np.min(y_indices), np.max(y_indices)
            else:
                
                x1, y1 = 0, 0
                x2, y2 = self.input_image_size[0], self.input_image_size[1]
                
            # 防止无效框 (宽高为0)
            if x2 <= x1: x2 = x1 + 1
            if y2 <= y1: y2 = y1 + 1
                
            boxes.append([x1, y1, x2, y2])
            
        boxes = torch.tensor(boxes, dtype=torch.float32, device=corr_map.device)
        return boxes

    def forward(self, img_feat, text_feat):
        # 1. 计算文本-图像相关性热力图
        corr_map = self.compute_correlation_map(img_feat, text_feat)

        
        boxes = self.generate_bounding_boxes_otsu(corr_map)

        # 3. Box Encoding 发送给 SAM
        boxes_reshaped = boxes.unsqueeze(1) # SAM expects [B, N, 4]
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=None,
            boxes=boxes_reshaped,
            masks=None
        )

        return sparse_embeddings, dense_embeddings, corr_map, boxes
