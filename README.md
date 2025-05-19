# 🧬 Yeast ORF Classifier with XGBoost & Optuna

A bioinformatics machine learning pipeline to predict Open Reading Frames (ORFs) in the *Saccharomyces cerevisiae* genome using engineered sequence features and a GPU-accelerated XGBoost model, optimized via Bayesian tuning with Optuna.

---

## Overview

This project focuses on distinguishing coding from non-coding ORFs in the yeast genome using interpretable machine learning. By combining biologically inspired features with XGBoost and Optuna, we aim to identify patterns that correlate with protein-coding potential.

---

## Dataset

- **Source**: [NCBI Genome - GCF_000146045.2](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/)
- **Inputs**: ORFs predicted from the reference *S. cerevisiae* genome.
- **Features** (examples):
  - `length_x`, `length_y`: Dimensions or coordinates of the ORF
  - `stop_percent`: Proportion of stop codons
  - `gc_content`: GC base content
  - `codon_usage`: Relative codon frequencies
- **Target**:
  - Binary label: `1` = known gene, `0` = putative/non-coding

---

## 🔬 Pipeline Steps

1. **Preprocessing**
   - Feature extraction & optional scaling
   - Label encoding and merging feature + label sets

2. **Train/Test Split**
   - Stratified 75/25 split to maintain class balance

3. **Hyperparameter Tuning**
   - Bayesian optimization via [Optuna](https://optuna.org/)
   - 5-fold cross-validation using ROC AUC as the metric

4. **Model Training**
   - XGBoost classifier with `device='cuda'` for GPU acceleration
   - Early stopping based on validation performance

5. **Evaluation**
   - Confusion Matrix
   - Precision / Recall / F1-score
   - ROC AUC

6. **Interpretability**
   - Feature importance plots
   - Partial Dependence Plots (PDP)

---

## ✅ Results

**Best ROC AUC (Optuna):** ~0.979  
**Test Set Metrics:**

| Metric     | Score |
|------------|-------|
| Precision  | 0.96  |
| Recall     | 0.98  |
| F1-score   | 0.97  |
| ROC AUC    | 0.979 |

**Top Features:**
- `length_y`
- `stop_percent`
- `length_x`

---

## Interpretability Insights

Partial Dependence Plots (PDP) revealed:

- `length_y` and `length_x` positively correlate with coding potential up to a plateau
- `stop_percent` shows a strong negative impact when > 0.005

These patterns reflect biological intuition and validate the model's interpretability.

---

## 🗂️ Project Structure
yeast_orf_classifier/
├── data/       # Raw and processed input data
├── src/        # Modular source code
├── outputs/    # Trained models, plots, reports
├── main.py     # Main execution script
├── README.md   # Project documentation
└── requirements.txt   # Python dependencies