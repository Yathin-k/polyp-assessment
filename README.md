# AI-Driven Automated Colorectal Polyp Assessment Using Advanced Computer Vision Techniques

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![Architecture](https://img.shields.io/badge/Model-ResUNet++-green.svg)](https://arxiv.org/abs/1911.07067)
[![Dataset](https://img.shields.io/badge/Roboflow-Kvasir--SEG-blueviolet.svg)](https://universe.roboflow.com/teste-mhypc/kvasir-seg-gouny)

An academic deep learning application for automated semantic segmentation of colorectal polyps in colonoscopy imagery using the published **ResUNet++** hybrid architecture ([Jha et al., 2019](https://arxiv.org/abs/1911.07067)). Includes a modular training pipeline with Google Colab GPU acceleration, rigorous scientific evaluation metrics, and a production-ready **Streamlit web application** for interactive clinical visualization.

---

## ⚠️ Academic & Clinical Disclaimer

> **IMPORTANT**: This software is an **academic research prototype** created for educational and scientific computer vision evaluation. It is **NOT** a certified medical device (FDA/CE-mark) and must **NOT** be used as a standalone diagnostic tool, to confirm malignancy, determine histology, or replace endoscopic examination by licensed medical professionals.

---

## 🔬 Architecture Overview: ResUNet++

ResUNet++ combines residual learning, attention mechanisms, and multi-scale context aggregation to overcome key challenges in colonoscopic image analysis (e.g., subtle boundaries, specular reflections, camera blur, and mucosal folds).

```
Input Image (3 × H × W)
         │
         ▼
    [Stem Block] ───────────────────────────────────────────┐ (skip 1: 32 ch)
         │ (stride 2 downsample)                             │
         ▼                                                   │
  [Encoder 1: ResConv + SE] ──────────────────────────┐      │ (skip 2: 64 ch)
         │ (stride 2)                                 │      │
         ▼                                            │      │
  [Encoder 2: ResConv + SE] ───────────────────┐      │      │ (skip 3: 128 ch)
         │ (stride 2)                          │      │      │
         ▼                                     │      │      │
  [Encoder 3: ResConv + SE]                   │      │      │
         │                                     │      │      │
         ▼                                     │      │      │
    [ASPP Bridge]                              │      │      │
         │ (rates: 1, 6, 12, 18)               │      │      │
         ▼                                     │      │      │
  [Decoder 3: Upsample + Attention Gate] ◄─────┘      │      │
         │                                            │      │
         ▼                                            │      │
  [Decoder 2: Upsample + Attention Gate] ◄────────────┘      │
         │                                                   │
         ▼                                                   │
  [Decoder 1: Upsample + Attention Gate] ◄───────────────────┘
         │
         ▼
    [Output ASPP] (rates: 1, 6, 12, 18)
         │
         ▼
 [1×1 Conv + Sigmoid Head]
         │
         ▼
Predicted Probability Map (1 × H × W)
```

### Key Architectural Components
1. **Residual Units**: Identity shortcut connections mitigate the vanishing gradient problem, allowing stable feature propagation across deep layers.
2. **Squeeze-and-Excitation (SE) Units**: Adaptive channel-wise feature recalibration that emphasizes polyp-relevant feature maps while suppressing background colon textures.
3. **Atrous Spatial Pyramid Pooling (ASPP)**: Dilated parallel convolutions at sampling rates $r \in \{1, 6, 12, 18\}$ to capture multi-scale context without loss of spatial resolution.
4. **Attention Gates (AG)**: Filter skip connection signals using gating signals from coarser decoder layers to suppress irrelevant mucosal backgrounds.

---

## 📁 Repository Structure

```
Polyp Assesment/
├── app.py                          # Interactive Streamlit Web Application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusion rules
├── configs/
│   └── resunetpp.yaml              # Hyperparameters and path configuration
├── notebooks/
│   └── train_resunetpp.ipynb       # 1-Click Google Colab GPU Training Notebook
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resunetpp.py            # Complete ResUNet++ implementation
│   │   └── unet.py                 # Baseline U-Net for comparative benchmarking
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # PyTorch Dataset with synchronized augmentations
│   │   ├── prepare_dataset.py      # Roboflow download & COCO-to-mask converter
│   │   ├── validate_dataset.py     # Dataset integrity & split balance reporting
│   │   └── split_dataset.py        # Reproducible 70/15/15 train/val/test splits
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py               # Combined BCE + Dice Loss
│   │   ├── metrics.py              # Dice, IoU, Precision, Recall, Specificity
│   │   └── train.py                # Modular training loop with checkpointing
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py             # Scientific benchmark evaluation script
│   │   └── visualize.py            # 4-panel prediction visualizer
│   └── inference/
│       ├── __init__.py
│       └── predict.py              # Reusable inference engine for web app
└── tests/
    ├── __init__.py
    ├── test_model.py               # Forward/backward pass & shape tests
    ├── test_dataset.py             # Synthetic pair & augmentation tests
    └── test_inference.py           # Predictor initialization & edge-case tests
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone or navigate to the project directory and install the requirements:

```bash
git clone https://github.com/Yathin-k/Capstone-DevOps.git
cd "Polyp Assesment"
pip install -r requirements.txt
```

---

### 2. Dataset Setup (Roboflow Universe)
The project utilizes the pre-annotated **Kvasir-SEG** dataset from Roboflow Universe:
- **URL**: [https://universe.roboflow.com/teste-mhypc/kvasir-seg-gouny](https://universe.roboflow.com/teste-mhypc/kvasir-seg-gouny)
- **Format**: `png-mask` (or `coco` with automatic polygon rasterization)

To download and organize the dataset locally:
```bash
# Option A: With environment variable
export ROBOFLOW_API_KEY="your_api_key_here"
python -m src.data.prepare_dataset

# Option B: Interactive prompt
python -m src.data.prepare_dataset
```

To validate the dataset integrity:
```bash
python -m src.data.validate_dataset --data-dir data/processed
```

---

### 3. Training on Google Colab (Recommended for GPU)
Because training convolutional networks like ResUNet++ benefits significantly from NVIDIA GPU acceleration:

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the notebook located at [notebooks/train_resunetpp.ipynb](file:///notebooks/train_resunetpp.ipynb).
3. Set runtime to GPU: **Runtime** → **Change runtime type** → **Hardware accelerator: T4 GPU**.
4. Run all cells sequentially. The notebook will:
   - Verify GPU availability
   - Download and standardize the dataset
   - Train ResUNet++ with hybrid BCE + Dice Loss and early stopping
   - Generate training loss/Dice convergence curves
   - Compute test-set benchmark metrics
   - Automatically trigger browser download of `resunetpp_best.pth`
5. Place the downloaded checkpoint into:
   ```
   artifacts/checkpoints/resunetpp_best.pth
   ```

---

### 4. Running the Streamlit Web Application
Once the checkpoint is placed in `artifacts/checkpoints/resunetpp_best.pth`:

```bash
streamlit run app.py
```

#### Features of the Web App:
- **Intuitive Upload**: Upload any colonoscopy image (JPG, PNG).
- **Interactive Thresholding**: Adjustable confidence slider ($0.10 - 0.90$).
- **Multi-view Visualization**:
  - Original colonoscopy frame
  - Predicted binary polyp mask
  - Alpha-blended clinical overlay
- **Quantitative Summary**:
  - Polyp detection status (Detected / No Polyp)
  - Affected mucosal surface area percentage
  - Mean segmentation confidence score
- **Export Capabilities**: 1-click download of generated mask and overlay images.
- **CPU Optimized**: Fast local inference (~150ms on standard laptop CPU).

---

### 5. Standalone Local Training & Evaluation (CLI)
You can also run training or evaluation directly via command line:

```bash
# Standalone Training
python -m src.training.train --config configs/resunetpp.yaml

# Benchmark Evaluation on Test Split
python -m src.evaluation.evaluate --checkpoint artifacts/checkpoints/resunetpp_best.pth --data-dir data/processed

# Generate Visualization Figures
python -m src.evaluation.visualize --checkpoint artifacts/checkpoints/resunetpp_best.pth --output-dir artifacts/figures
```

---

## 📊 Scientific Evaluation Metrics

Segmentation performance is quantitatively evaluated across 5 core metrics:

| Metric | Formula | Clinical Meaning |
| :--- | :--- | :--- |
| **Dice Similarity Coefficient (DSC / F1)** | $\frac{2 |P \cap G|}{|P| + |G|}$ | Overall boundary overlap accuracy between prediction $P$ and ground truth $G$. |
| **Mean Intersection over Union (mIoU)** | $\frac{|P \cap G|}{|P \cup G|}$ | Standard Jaccard index measuring area overlap. |
| **Recall / Sensitivity** | $\frac{TP}{TP + FN}$ | Critical for minimizing missed polyps (false negatives). |
| **Precision (PPV)** | $\frac{TP}{TP + FP}$ | Proportion of segmented pixels that are true polyp tissue. |
| **Specificity (TNR)** | $\frac{TN}{TN + FP}$ | Ability to correctly reject normal healthy mucosal tissue. |

---

## 🧪 Automated Testing Suite

The repository includes a comprehensive unit and integration test suite:

```bash
# Run all unit tests
pytest tests/ -v
```

- [tests/test_model.py](file:///tests/test_model.py): Verifies forward/backward passes, tensor output shapes, gradient flow, and parameter count sanity.
- [tests/test_dataset.py](file:///tests/test_dataset.py): Validates image-mask synchronization, binarization, data augmentations, and batch loading.
- [tests/test_inference.py](file:///tests/test_inference.py): Validates inference engine initialization, prediction formats, threshold adjustments, and graceful error handling.

---

## 📚 References
1. **ResUNet++ Paper**: Jha, D., Smedsrud, P. H., Riegler, M. A., Johansen, D., De Lange, T., & Halvorsen, P. (2019). *ResUNet++: An Advanced Architecture for Medical Image Segmentation*. [arXiv:1911.07067](https://arxiv.org/abs/1911.07067).
2. **Kvasir-SEG Dataset**: Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D., & Johansen, H. D. (2020). *Kvasir-SEG: A Segmented Polyp Dataset*. In MultiMedia Modeling (MMM).
3. **Roboflow Universe Project**: [Kvasir-SEG Dataset on Roboflow](https://universe.roboflow.com/teste-mhypc/kvasir-seg-gouny).
