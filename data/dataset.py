import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import clip
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ClipSamDataset(Dataset):
    def __init__(self,json_path,root_dir,img_size=256,is_train=True, prompt_mode="personalized_test"):
        """
        Args:
            json_path: 数据索引文件路径
            root_dir: 数据根目录
            img_size: 统一分辨率 256
            is_train: 是否为训练模式
            prompt_mode: 提示词生成模式
                1. "random_detailed": 随机模版 + 详细提示
                2. "single_simple":   单一模版 + 简单提示
                3. "single_detailed": 单一模版 + 详细提示
                4. "basic":           仅器官名称 (最弱基线)
                5. "personalized_train" 真实场景下的个性化输入训练
                6. "personalized_test"
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.img_size = img_size
        self.prompt_mode = prompt_mode

        # 1. 加载索引
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 2. 基础调整 (Base Resize) -> 统一到 256
        self.base_resize = A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
        ])

        # === 核心修改：提示词模版定义 ===

        # A. 随机模版库 (用于 random_simple 和 random_detailed)
        # {target} 将会被替换为 organ (简单) 或 description (详细)

        # 确保每个模版都包含 {modality}
        self.RANDOM_TEMPLATES = [
            # 1. 极简陈述句 (高度契合 CLIP 预训练分布)
            "A {modality} of a {target}.",
            "A photo of a {target} in a {modality} scan.",
            
            # 2. 存在性描述句 
            "There is a {target} in this {modality}.",
            "This {modality} contains a {target}.",
            
            # 3. 医学影像专业描述句
            "A medical {modality} showing the anatomical structure of the {target}.",
            "The {target} is clearly visualized in this {modality} scan.",
            "Cross-sectional {modality} capturing the {target}.",
            "Radiological {modality} presenting the {target} region."
        ]

        # B. 单一模版 (用于 single_simple 和 single_detailed)
        # 最原汁原味的极简描述结构
        self.SINGLE_TEMPLATE = "A {modality} of a {target}."

        # 3. 数据增强配置 (保持不变)
        if is_train:
            # 预计算 Dropout 的尺寸范围
            h_min, h_max = int(img_size * 0.05), int(img_size * 0.2)
            w_min, w_max = int(img_size * 0.05), int(img_size * 0.2)

            self.aug_transform = A.Compose([
                # =========================
                # 1. Geometry (几何变换)
                # =========================
                A.Compose([
                    # [1.3.1 适配] Affine 使用 mode, cval, cval_mask
                    A.Affine(
                        scale=(0.85, 1.25),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-30, 30),
                        shear=(-15, 15),
                        interpolation=cv2.INTER_LINEAR,
                        mode=cv2.BORDER_CONSTANT,  # 1.3.1 参数名: mode
                        cval=0,  # 1.3.1 参数名: cval
                        cval_mask=0,  # 1.3.1 参数名: cval_mask
                        p=0.5
                    ),

                    A.OneOf([
                        # [1.3.1 适配] Elastic/Grid 等使用 border_mode, value, mask_value
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            alpha_affine=0,  # 1.3.1 显式设为 None 或移除
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0
                        ),
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0
                        ),
                        A.OpticalDistortion(
                            distort_limit=0.05,
                            shift_limit=0.05,  # 1.3.1 需要 shift_limit
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=0,
                            p=1.0
                        ),
                    ], p=0.2),
                ]),

                # =========================
                # 2. Low Quality (低质量模拟)
                # =========================
                A.OneOf([
                    # [1.3.1 适配] 使用 scale_min/scale_max 代替 scale
                    A.Downscale(scale_min=0.75, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=1.0),

                    # [1.3.1 适配] 使用 quality_lower/quality_upper
                    A.ImageCompression(quality_lower=50, quality_upper=100, p=1.0),
                ], p=0.2),

                # =========================
                # 3. Noise & Blur (噪声与模糊)
                # =========================
                A.OneOf([
                    # [1.3.1 适配] 类名为 GaussNoise (非 GaussianNoise)，参数为 var_limit
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),

                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                ], p=0.2),

                # =========================
                # 4. Intensity (强度)
                # =========================
                A.Compose([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    ),
                    A.RandomGamma(gamma_limit=(70, 150), p=0.5),
                    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
                ], p=0.5),

                # =========================
                # 5. Regularization (正则化)
                # =========================
                # [1.3.1 适配] 使用 max_holes, fill_value, mask_fill_value
                A.CoarseDropout(
                    max_holes=4,  # 固定孔洞数量上限
                    min_holes=4,  # 固定孔洞数量下限 (等效于 num_holes=4)
                    max_height=h_max,
                    max_width=w_max,
                    min_height=h_min,
                    min_width=w_min,
                    fill_value=0,  # 1.3.1 参数名
                    mask_fill_value=0,  # 1.3.1 参数名
                    p=0.2
                ),
            ], additional_targets={"mask": "mask"})
        else:
            self.aug_transform = None

        # 4. CLIP 均值方差
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def _generate_prompt(self, item):
        """
        实现 6 种 Prompt 组合逻辑:
        - basic: 仅器官名称 (最弱基线)
        - personalized: 随机模拟真实医生的个性化输入 (语音、病历、缩写、错别字、学术术语)
        - single_simple / random_simple: 模板 + 器官名称
        - single_detailed / random_detailed: 模板 + 器官名称 + 专家级详细描述
        """
        # 1. 提取基础信息
        organ = item.get('organ', 'object').lower().strip()

        # ================= 🌟 模式 1：Basic 纯词汇拦截 =================
        if "basic" in self.prompt_mode:
            return organ

        # ================= 🌟 模式 2：Personalized_test 个性化临床输入测试 =================
        if "personalized_test" in self.prompt_mode:
            # 1. 模拟快速打字的漏键错误 (Typo)
            typo_organ = organ
            if len(organ) > 4:
                idx = random.randint(1, len(organ) - 2)
                typo_organ = organ[:idx] + organ[idx + 1:]  # 随机删掉中间的一个字母

            # 2. 模拟解剖学/学术同义词 (Medical Synonym - 测试集专属)
            # 替换为比训练集更贴近“影像学报告”的真实高级术语
            synonyms = {
                "liver": "hepatic contour",          # 肝脏轮廓
                "kidney": "renal parenchyma",        # 肾实质（临床极其常用）
                "lung": "pulmonary fields",          # 肺野
                "heart": "myocardium",               # 心肌
                "brain tumor": "intracranial mass",  # 颅内肿块（影像科下诊断前的标准叫法）
                "stomach": "gastric wall"            # 胃壁
            }
            academic_term = synonyms.get(organ, organ)

            # 3. 真实场景语料库 (极致拟真版 OOD)
            personalized_pool = [
                f"Could you outline the {organ} for me?",  
                # 场景 A 对话式：语音听写中最自然的祈使句，医生更习惯说 outline (勾勒) 而不是 highlight。

                f"Prior imaging reviewed. No interval change. Please segment the {organ} to confirm dimensions.", 
                # 场景 B 电子病历噪音："Prior imaging reviewed. No interval change." (已复阅既往影像，无近期变化) 是全球影像科报告里最高频的“套话噪音”，极具真实感。

                f"{organ} vol",  
                # 场景 C 极简缩写：医生在急诊打字时，相比于要个 "seg" (分割)，他们更真实的目的是要 "vol" (体积，volume的缩写)。

                f"Segment the {typo_organ}.",  
                # 场景 D 拼写错误：保留你的逻辑，最经典的漏键错误。

                f"Evaluate the {academic_term}."  
                # 场景 E 术语偏好：高年资医生通常会说“评估(evaluate)某某实质”，而不是直接下达“分割”指令。
            ]

            # 随机返回一种临床输入形式
            return random.choice(personalized_pool)
        
        
        # # ================= 🌟 模式 3：Personalized_train 个性化临床输入训练 =================
        if "personalized_train" in self.prompt_mode:
            # 1. 模拟快速打字的漏键错误 (Typo)
            typo_organ = organ
            if len(organ) > 4:
                idx = random.randint(1, len(organ) - 2)
                typo_organ = organ[:idx] + organ[idx + 1:]  # 随机删掉中间的一个字母

            # 2. 模拟解剖学/学术同义词 (Medical Synonym)
            synonyms = {
                "liver": "hepatic parenchyma",
                "kidney": "renal tissue",
                "lung": "pulmonary area",
                "heart": "cardiac silhouette",
                "brain tumor": "cerebral lesion",
                "stomach": "gastric region"
            }
            academic_term = synonyms.get(organ, organ)

            
            
           # 3. 真实场景语料库 (极度扩充版)
            personalized_pool = [
            # ================= 场景 A: 强指令/学术操作 (Imperative & Academic) =================
            f"Extract the {organ} from this image.",
            f"Delineate the contours of the {organ}.",
            f"Generate a segmentation mask for the {organ} region.",
            f"Please isolate the {organ} structure.",
            
            # ================= 场景 B: 语音助手/口语化对话 (Conversational) =================
            f"Could you please segment the {organ} for me?",
            f"Show me the {organ} in this scan.",
            f"I need to take a look at the {organ}, please highlight it.",
            f"Can you find the {organ} here?",
            f"Help me locate the {organ}.",
            
            # ================= 场景 C: 电子病历/带噪上下文 (EHR Narrative Noise) =================
            # 核心目的：训练模型在长句子中“大海捞针”提取目标
            f"Patient reports localized pain. Reviewing the scan now. Please isolate the {organ} to check for abnormalities.",
            f"Routine screening. Focus on the {organ} and delineate its boundaries.",
            f"Evaluate the scan for potential lesions. Segment the {organ} for volume measurement.",
            f"History of trauma. Extract the {organ} mask to assess potential damage.",
            
            # ================= 场景 D: 急诊极简/医生黑话 (Rushed & Shorthand) =================
            # 核心目的：测试模型在缺乏语法结构时的关键词抓取能力
            f"need {organ} mask",
            f"{organ} seg",
            f"loc {organ}",
            f"get {organ}",
            
            # ================= 场景 E: 术语替换与拼写错误 (Terminology & Typos) =================
            f"Segment the {typo_organ}.",
            f"find the {typo_organ} pls",
            f"Delineate the {academic_term}."
        ]

            # 随机返回一种临床输入形式
            return random.choice(personalized_pool)
        raw_modality = item.get('modality', 'medical image').strip().lower()

        # ================= 确定性模态全称映射 (过滤工程后缀) =================
        # 将各种带有工程处理痕迹的名称 (如 2.5D, PseudoRGB) 强制映射回纯净的临床模态全称
        
        if raw_modality in ['ct', 'ct scan', 'computed tomography', 'ct_2.5d']:
            modality = "computerized tomography"  
            
        elif raw_modality in ['mri', 'mr', 'magnetic resonance', 'mri_pseudorgb', 'mri (spatial 2.5d)']:
            modality = "magnetic resonance imaging"
            
        elif raw_modality in ['us', 'ultrasound', 'echo', 'sonogram']:
            modality = "ultrasound"
            
        elif raw_modality in ['xray', 'x-ray', 'cxr', 'radiograph', 'chest x-ray']:
            modality = "radiograph"
            
        
            
        else:
            # 兜底逻辑
            modality = raw_modality if raw_modality else "medical image"
        

        # 提取专家描述，如果没有则兜底为空字符串
        description = item.get('description', '').strip()

        # 文本清洗：确保描述首字母大写，且以句号结尾，使其更符合自然语言习惯
        if description:
            description = description[0].upper() + description[1:]
            if not description.endswith('.'):
                description += '.'

        # 2. 确定句式模版 (Template Variability: Single vs. Random)
        if "random" in self.prompt_mode:
            template = random.choice(self.RANDOM_TEMPLATES)
        elif "single" in self.prompt_mode:
            template = self.SINGLE_TEMPLATE
        else:
            template = self.SINGLE_TEMPLATE  # 默认兜底

        # 3. 填入基础词汇生成 Base Prompt
        base_prompt = template.format(target=organ, modality=modality)

        # 4. 确定语义丰富度 (Semantic Granularity: Simple vs. Detailed)
        if "detailed" in self.prompt_mode:
            # 详细模式：将基础句式与专家描述自然拼接
            if description and description.lower() != organ.lower() + '.':
                final_prompt = f"{base_prompt} {description}"
            else:
                final_prompt = base_prompt  # 若无有效详述，兜底为简单提示
        else:
            # 简单模式：仅使用基础句式
            final_prompt = base_prompt

        return final_prompt.strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 路径拼接
        img_path = os.path.join(self.root_dir, item['img_path'].replace("\\", "/"))
        mask_path = os.path.join(self.root_dir, item['mask_path'].replace("\\", "/"))

        try:
            # A. 数据加载
            if img_path.endswith('.npy'):
                image = np.load(img_path)
                mask = np.load(mask_path)
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif image.shape[2] == 1:
                    image = np.concatenate([image] * 3, axis=-1)
            else:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, 0)

            # 强制类型转换 (uint8)
            if image.dtype != np.uint8:
                if image.max() <= 1.5:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            mask = (mask > 0).astype(np.uint8)

            # B. 基础 Resize
            base_aug = self.base_resize(image=image, mask=mask)
            image = base_aug['image']
            mask = base_aug['mask']

            # C. 数据增强
            if self.is_train and self.aug_transform:
                augmented = self.aug_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            # D. 转 Tensor & 归一化
            img_tensor_base = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

            # E. 生成 CLIP 输入
            img_tensor_clip = (img_tensor_base - self.clip_mean) / self.clip_std

            # F. 文本处理 (使用新的 4 种模式逻辑)
            # ==========================================
            prompt_text = self._generate_prompt(item)
            # ==========================================

            text_token = clip.tokenize(prompt_text, truncate=True).squeeze(0)

            # G. 元数据
            has_object = 1.0 if mask_tensor.max() > 0 else 0.0
            organ_name = item.get('organ', 'Unknown')
            source = item.get('source', 'Unknown')

            return {
                "image_clip": img_tensor_clip,
                "gt_mask": mask_tensor,
                "has_object": torch.tensor(has_object),
                "text_token": text_token,
                "prompt_text": prompt_text,
                "source": source,
                "organ": organ_name
            }

        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}. Skipping...")
            return self.__getitem__(random.randint(0, len(self.data) - 1))