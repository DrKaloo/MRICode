# Robustness of 3D CNN Alzheimer's Classification to Age Distribution Shift

Investigating how age distribution shift affects CN/MCI/AD classification performance in 3D CNNs trained on structural MRI, and evaluating mitigation strategies to improve worst-age-group performance.

-----------

## Overview

Deep learning models for Alzheimer's disease classification often report high aggregate accuracy but fail to generalise across age groups and datasets. This project systematically measures and mitigates age distribution shift using:

- **3-class classification**: CN (Cognitively Normal) / MCI (Mild Cognitive Impairment) / AD (Alzheimer's Disease)
- **Age-stratified evaluation**: Performance by age bin (<70, 70-80, >80) with bootstrap confidence intervals
- **Cross-cohort validation**: Train on OASIS, test on ADNI
- **Mitigation strategies**: Age-matched sampling, group-robust optimisation, adversarial age-invariance

-----------

## Research Questions

1. How does age distribution shift affect CN/MCI/AD classification performance?
2. Which mitigation strategy most improves worst-age-group performance?

-----------

## Project Structure

```
├── data/
│   ├── raw/                    # Raw OASIS scans (OAS1 + OAS2)
│   ├── processed_128/          # Preprocessed 128³ volumes
│   ├── splits/validated/       # Train/val/test CSVs
│   └── metadata.csv
│
├── results/
│   ├── best.pt                 # Model checkpoint
│   ├── eval_summary.json       # Overall metrics
│   ├── test_age_stratified.csv # Age-bin performance
│   └── test_bootstrap_ci_*.csv # Confidence intervals
│
├── preprocess_all.py           # OASIS preprocessing
├── preprocess_ADNI.py          # ADNI preprocessing  
├── data_splits.py              # Subject-level splitting
├── dataset.py                  # PyTorch Dataset
├── resnet3d.py                 # 3D ResNet-18 architecture
├── train.py                    # Training with EMA, focal loss
├── evaluate.py                 # Evaluation + bootstrap CIs
└── demographic_analysis.py     # Age-stratified analysis
```

-----------

## Usage

### 1. Preprocess

```bash
python preprocess_all.py    # OASIS (OAS1 + OAS2)
python preprocess_ADNI.py   # ADNI
```

**Pipeline:** Brain masking → intensity clipping (1st-99th percentile) → z-score normalisation → crop to bounding box → resample to 128³

### 2. Create Splits

```bash
python data_splits.py --resolution 128
```

Subject-level splits stratified by diagnosis. One scan per subject to prevent leakage.

### 3. Train

```bash
python train.py \
    --backbone r3d_18 \
    --epochs 60 \
    --batch_size 2 \
    --lr 3e-4 \
    --focal_gamma 1.5 \
    --label_smoothing 0.05 \
    --ema
```

### 4. Evaluate

```bash
python evaluate.py --ckpt results/best.pt
```

Outputs: `eval_summary.json`, `test_age_stratified.csv`, bootstrap CIs

-----------

## Datasets

| Dataset | Description | Subjects |
|---------|-------------|----------|
| **OASIS-1** | Cross-sectional T1 MRI,         ages 18-96 | 416 |
| **OASIS-2** | Longitudinal T1 MRI,            ages 60-96 | 150 |
| **ADNI** | External validation cohort | — |

**Label mapping:** CDR 0 → CN, CDR 0.5 → MCI, CDR 1+ → AD

-----------

## Model

- **Architecture:** 3D ResNet-18 (torchvision video backbone)
- **Input:** Single-channel 96³ patches from 128³ volumes
- **Training:** Focal loss (γ=1.5), label smoothing (0.05), EMA, AdamW
- **Evaluation:** Macro-F1, age-stratified metrics, bootstrap CIs

-----------

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Macro-F1 | Primary metric (handles class imbalance) |
| Balanced Accuracy | Mean per-class recall |
| AUC-ROC | Discrimination ability |
| ECE | Calibration error |
| Age-bin TPR/FNR | Fairness across age groups |

**Primary success criterion:** Statistically significant reduction in worst-age-bin macro-F1 gap under at least one mitigation strategy.
