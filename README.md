
# CTF-SAM: A Lightweight CLIP-Driven Tri-Modal Fusion Framework for Robust Text-Guided Medical Image Segmentation

This is the official implementation of the paper: **"CTF-SAM: A Lightweight CLIP-Driven Tri-Modal Fusion Framework for Robust Text-Guided Medical Image Segmentation."**

**CTF-SAM** is a high-efficiency model designed for text-guided medical image segmentation. It utilizes a tri-modal fusion strategy to integrate image, text, and prompt information. To significantly reduce computational overhead and improve inference speed, we removed the heavy SAM image encoder and repurposed the CLIP visual backbone for image feature extraction, achieving precise zero-shot and few-shot segmentation.

## 🌟 Features

* **Official Implementation**: Provides the complete pipeline for training, validation, and inference.
* **Lightweight Architecture**: Replaces the redundant SAM image encoder by reusing the CLIP visual backbone, drastically reducing GPU memory consumption.
* **Two-Stage Training**: Supports phased unfreezing (Stage 1: Freeze CLIP backbone; Stage 2: Unfreeze visual components) to maintain stable text-feature alignment.
* **Mixed Precision & Acceleration**: Native support for BF16/FP16 Automatic Mixed Precision (AMP) and gradient accumulation for various hardware setups.
* **EMA Support**: Built-in Exponential Moving Average (EMA) via `torch_ema` to enhance model generalization and robustness.
* **Comprehensive Evaluation**: Supports both Dice Score and NSD (Normalized Surface Distance), generating macro-average reports and high-quality visualizations.

---

## 🛠️ Installation

We provide a comprehensive configuration file for environment setup. We recommend using Conda for environment management.

```bash
# 1. Clone the repository
# git clone https://github.com/lihang-cat/CTF-SAM.git
# cd CTF-SAM

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate CTF-SAM
```

---

## 📂 Data & Checkpoints

We provide pre-processed datasets (JSON format) and pre-trained weights for reproducibility.

🔗 **[Download Dataset & Weights from Google Drive](https://drive.google.com/drive/folders/1aGJB8xSI63jGxxO_G4mn0BQ39voZrNKd?usp=sharing)**

### Recommended Directory Structure
After downloading, organize your files as follows (or update the paths in `CONFIG`):

```text
CTF-SAM-main/
├── checkpoints/
│   ├── sam_vit_b_01ec64.pth          # Original SAM weights for initialization
│   └── CTF-SAM-personalized_train/   # Training outputs saved here
├── data/
│   ├── dataset.py                    # Dataset class implementation
│   └── processed/
│       ├── total_train.json          # Training set index
│       ├── total_val.json            # Validation set index
│       └── total_test.json           # Test set index
├── data_root/                        # (Default: /root/sj-tmp/Med-CLIP-SAM)
│   └── ... (Extracted images and masks)
├── models/                           # Model architecture definitions
├── utils/                            # Loss functions and utilities
├── train.py                          # Training script
└── eval.py                           # Evaluation script
```

> **Note**: The default `data_root` is set to `/root/sj-tmp/Med-CLIP-SAM` in the scripts. Please update this to your local path in `train.py` and `eval.py`.

---

## 🚀 Training

To start training the model, run:

```bash
python train.py
```

**Key Configurations (in `train.py` `CONFIG`)**:
* `batch_size` / `accum_iter`: Adjusted for GPU memory (Default: `bs=42`, `accum_iter=3`).
* `unfreeze_epoch`: Defines when to unfreeze the CLIP backbone (Default: epoch 5).
* `use_ema`: Enables Exponential Moving Average for weight smoothing.
* **W&B Integration**: Integrated with `wandb` for experiment tracking. Ensure you are logged in to your account.

---

## 📊 Evaluation & Visualization

After training or downloading the pre-trained weights (e.g., `best_model.pth`), evaluate performance using:

```bash
python eval.py
```

**Evaluation Functionality**:
1. **Quantitative Metrics**: Calculates Dice Score and NSD (Tolerance=2.0) for every dataset and organ.
2. **Report Generation**: Prints a global macro-average report in the terminal and exports results to `clipsamnet_macro_results.csv`.
3. **Qualitative Analysis**: Saves visual samples (Image with prompt, Ground Truth, and Prediction) to `vis_results_v1/normal_samples/`.
