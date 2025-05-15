# Yeast ORF Classifier with XGBoost and Optuna

A machine learning pipeline for predicting Open Reading Frames (ORFs) in the *Saccharomyces cerevisiae* genome using engineered sequence features and XGBoost, optimized via Bayesian tuning (Optuna).

---

## Project Overview

This project aims to distinguish between true coding ORFs and non-coding sequences based on genomic features. It uses a GPU-accelerated XGBoost model for classification.

---

## Dataset

**Input**: ORFs predicted from the yeast genome. Genome can be obtained via: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/

**Features**:

* `length_x`: Length of ORF in the X dimension
* `length_y`: Length of ORF in the Y dimension
* `stop_percent`: Proportion of stop codons
* `gc_content`, `codon_usage`, etc. (optional additional features)

**Target**: Binary label indicating if an ORF is a known gene (`1`) or not (`0`).

---

## Pipeline Components

1. **Preprocessing**: Feature scaling, missing value imputation (if needed).
2. **Train-Test Split**: Stratified 75/25 split.
3. **Bayesian Optimization**: XGBoost hyperparameters tuned with Optuna.
4. **Model Training**: GPU-accelerated XGBoost (`device="cuda"`).
5. **Evaluation**: Confusion Matrix, Classification report, ROC AUC.
6. **Interpretability**: Feature importance.

---

## Results

**Best AUC (Optuna)**: \~0.979
**Test Set Performance**:

* Precision: 0.96
* Recall: 0.98
* F1-score: 0.97
* ROC AUC: 0.979

**Top Features**:

* `length_y`
* `stop_percent`
* `length_x`

---

## Interpretability

Partial dependence plots showed plateauing effects of `length_y` and `length_x`, and a sharp drop for `stop_percent` below \~0.005.

---

## Folder Structure

```
yeast_orf_classifier/
├── data/                   # Raw and processed data
├── notebooks/              # EDA and training notebooks
├── src/                    # Core source code
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── interpretability.py
├── outputs/                # Saved models, plots, and metrics
├── main.py                 # Main training script
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

---
