# Bank Marketing Term Deposit Propensity Modeling

![R](https://img.shields.io/badge/R-%3E%3D%204.2-276DC3?logo=r&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-UCI%20Bank%20Marketing%20(N%3D45%2C211)-informational)

End-to-end binary classification and probability calibration pipeline predicting client subscription to retail bank term deposits on the full **UCI Bank Marketing dataset** ($N = 45,211$).

---

## 1. Business Context & Problem Statement

Outbound telemarketing campaigns face an asymmetric operational trade-off:
- **False Positives (FP)** waste call-center advisor capacity and generate customer fatigue.
- **False Negatives (FN)** represent lost deposit inflows and missed cross-selling opportunities.

This repository benchmarks interpretable and ensemble statistical learning models—**Cost-Complexity Pruned CART**, **Binomial Logistic Regression**, and two **Random Forest** engines (`randomForest` and C++-accelerated `ranger`)—to rank leads by conversion propensity and calibrate actionable probability cutoffs.

---

## 2. Dataset & Methodological Safeguards

The pipeline uses the full Portuguese banking institution dataset (`bank-full.csv`, Moro et al., 2014):
- **Sample Size:** $45,211$ client contact records across $16$ socio-demographic, financial, and campaign attributes (zero missing values).
- **Target Imbalance:** Only **$11.70\%$** ($5,289 / 45,211$) subscribed to a term deposit (`y = "yes"`), yielding a **No Information Rate (NIR) of $88.30\%$**.
- **Target Leakage Prevention (`duration` excluded):** The raw `bank-full.csv` file includes `duration` (call length in seconds). Because call duration is unknown prior to dialing—and `duration = 0` deterministically implies `y = "no"`—including it leaks post-hoc outcome information. `duration` is strictly dropped prior to training so that scores reflect genuine **pre-call conversion propensity**.

---

## 3. Repository Structure

```text
bank-marketing-propensity-modeling/
├── README.md
├── LICENSE
├── .gitignore
├── install_packages.R        # CRAN dependency setup
├── main.R                    # CLI pipeline runner
├── R/
│   ├── data_loader.R         # Automated UCI download, schema checks & leakage removal
│   ├── eda.R                 # Exploratory data visualizations
│   ├── models.R              # Pruned CART, Logistic Regression, randomForest & ranger
│   └── evaluation.R          # Confusion matrices, ROC-AUC & Youden threshold tuning
├── tests/
│   └── test_pipeline.R       # testthat unit tests
├── data/                     # Git-ignored local dataset cache
└── assets/                   # Generated figures & benchmark CSV
