import torch
import torch.nn as nn
import math


class LoRALinear(nn.Linear):
    """
    一个带 LoRA 分支的 Linear 层，用于替换标准的 nn.Linear
    """

    def __init__(self, in_features, out_features, r=4, lora_alpha=1, lora_dropout=0.0, **kwargs):
        # 初始化父类 nn.Linear
        super().__init__(in_features, out_features, **kwargs)

        # LoRA 超参数
        self.r = r
        self.lora_alpha = lora_alpha
        # Scaling factor: paper 建议 alpha/r
        self.scaling = self.lora_alpha / self.r

        # LoRA 矩阵 A (降维) 和 B (升维)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_A = None
            self.lora_B = None

        # 冻结原始权重 W (frozen by default in logic, but explicit here helps)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.reset_parameters_lora()

    def reset_parameters_lora(self):
        if hasattr(self, 'lora_A'):
            # 初始化策略：A 用 Kaiming Uniform，B 用 0
            # 这样初始状态下，LoRA 输出为 0，完全等价于预训练模型
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 1. 计算原始 Linear 输出 (Frozen W)
        result = super().forward(x)

        # 2. 加上 LoRA 分支 (Trainable A & B)
        if self.r > 0:
            # result = Wx + (B*A)x * scaling
            lora_out = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            result += lora_out

        return result

    @classmethod
    def from_linear(cls, linear_layer, r=4, lora_alpha=1, lora_dropout=0.0):
        """
        工厂方法：将一个现有的 nn.Linear 转换为 LoRALinear，并复制权重
        """
        lora_layer = cls(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=(linear_layer.bias is not None)
        )

        # 复制原始权重
        lora_layer.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            lora_layer.bias.data = linear_layer.bias.data.clone()

        # 确保原始权重不更新
        lora_layer.weight.requires_grad = False
        if lora_layer.bias is not None:
            lora_layer.bias.requires_grad = False

        return lora_layer


def inject_lora_to_clip(clip_model, r=4, lora_alpha=4, target_layers=['c_fc', 'c_proj', 'out_proj']):
    """
    遍历 CLIP 模型，将指定的 Linear 层替换为 LoRALinear

    Args:
        clip_model: 你的 CLIPBackbone.clip_model
        r: LoRA Rank (医学图像推荐 4 或 8)
        target_layers: 要注入 LoRA 的层名称后缀。
            OpenAI CLIP 结构中:
            - visual.transformer.resblocks[i].attn.out_proj (Attention 输出)
            - visual.transformer.resblocks[i].mlp.c_fc    (MLP 第一层)
            - visual.transformer.resblocks[i].mlp.c_proj  (MLP 第二层)
            注: OpenAI CLIP 的 QKV 是合并在 in_proj_weight 里的，很难替换，
            所以微调 MLP 和 Attention Output 是最稳妥的策略。
    """

    # 递归函数用于查找和替换
    def replace_layers(model, prefix=""):
        for name, child in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # 只有 Vision Transformer 部分需要 LoRA，Text Encoder 可选 (通常不动)
            # if "visual" not in full_name: continue

            # 判断是否是目标层
            if isinstance(child, nn.Linear) and any(t in name for t in target_layers):
                print(f"Injecting LoRA (r={r}) -> {full_name}")

                # 替换为 LoRALinear
                new_layer = LoRALinear.from_linear(child, r=r, lora_alpha=lora_alpha)
                setattr(model, name, new_layer)
            else:
                # 递归
                replace_layers(child, full_name)

    print(">>> Start LoRA Injection...")
    replace_layers(clip_model)
    print(">>> LoRA Injection Complete.")

    return clip_model


def mark_only_lora_as_trainable(model):
    """
    冻结非 LoRA 参数，只激活 lora_A 和 lora_B
    """
    for n, p in model.named_parameters():
        if 'lora_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # 统计参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} / {total_params} ({trainable_params / total_params:.2%})")