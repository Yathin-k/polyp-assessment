# Implementation Plan — ResUNet++ Polyp Segmentation Project

## Phase 1 — Workspace & Research Inspection ✅

### Findings
- **Workspace**: Empty directory at `c:\Users\Yathin\Downloads\Polyp Assesment`
- **Paper**: Jha et al., "ResUNet++: An Advanced Architecture for Medical Image Segmentation" (arXiv: 1911.07067, 2019)
- **Architecture verified**: Stem → 3 Encoder blocks (ResidualConv + SE) → ASPP bridge → 3 Decoder blocks (Attention Gates + Upsample + ResidualConv) → ASPP → 1×1 Conv → Sigmoid
- **Filter progression**: [32, 64, 128, 256, 512] (from reference implementation)
- **Reference repo**: `rishikksh20/ResUnet` (PyTorch) — verified structure
- **Dataset**: Roboflow `teste-mhypc/kvasir-seg-gouny` — Instance Segmentation project, supports `png-mask` and `coco` export formats. Requires API key to download.
- **No GPU locally**: Training must go through Google Colab

### Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| No local GPU | Colab notebook with GPU, CPU fallback for smoke tests |
| Dataset requires API key | Provide clear instructions, never hardcode keys |
| Dataset may be polygon-based | Support COCO JSON → binary mask conversion |
| Checkpoint not available at frontend time | Graceful error + clear instructions |

---

## Phase 2 — Core Model & Data Pipeline

### Files to Create
1. `src/models/resunetpp.py` — Full ResUNet++ (Stem, ResidualConv+SE, ASPP, AttentionGate, Decoder, Output ASPP)
2. `src/models/unet.py` — Standard U-Net baseline
3. `src/data/dataset.py` — PyTorch Dataset for image+mask pairs
4. `src/data/prepare_dataset.py` — Download via Roboflow API + convert to binary masks
5. `src/data/validate_dataset.py` — Validation report generator
6. `src/data/split_dataset.py` — Reproducible train/val/test splits
7. `tests/test_model.py` — Model smoke tests
8. `tests/test_dataset.py` — Dataset pipeline tests
9. `tests/test_inference.py` — Inference pipeline tests

---

## Phase 3 — Training & Evaluation

### Files to Create
1. `src/training/train.py` — Reusable training loop
2. `src/training/losses.py` — DiceLoss + BCE combo
3. `src/training/metrics.py` — Dice, IoU, Precision, Recall, Specificity
4. `src/evaluation/evaluate.py` — Test-set evaluation
5. `src/evaluation/visualize.py` — Visualization utilities
6. `configs/resunetpp.yaml` — Training configuration
7. `notebooks/train_resunetpp.ipynb` — Colab notebook

---

## Phase 4 — Frontend & Inference

### Files to Create
1. `src/inference/predict.py` — Reusable inference module
2. `app.py` — Streamlit application

---

## Phase 5 — Integration & Handover

### Files to Create
1. `requirements.txt`
2. `.gitignore`
3. `README.md`

---

## Architecture Summary (ResUNet++)

```
Input (3×H×W)
    │
    ▼
[Stem Block] ─────────────────────────────────┐ (skip1)
    │ (stride 2 downsample)                    │
    ▼                                          │
[Encoder 1: ResConv+SE] ──────────────────┐    │ (skip2)
    │ (stride 2)                           │    │
    ▼                                      │    │
[Encoder 2: ResConv+SE] ─────────────┐    │    │ (skip3)
    │ (stride 2)                      │    │    │
    ▼                                 │    │    │
[Encoder 3: ResConv+SE]              │    │    │
    │                                 │    │    │
    ▼                                 │    │    │
[ASPP Bridge]                        │    │    │
    │                                 │    │    │
    ▼                                 │    │    │
[Decoder 3: Upsample + Attn(skip3)] ◄┘    │    │
    │                                      │    │
    ▼                                      │    │
[Decoder 2: Upsample + Attn(skip2)] ◄─────┘    │
    │                                           │
    ▼                                           │
[Decoder 1: Upsample + Attn(skip1)] ◄──────────┘
    │
    ▼
[Output ASPP]
    │
    ▼
[1×1 Conv → Sigmoid]
    │
    ▼
Output (1×H×W)
```
