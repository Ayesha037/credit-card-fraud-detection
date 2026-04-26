# Credit Card Fraud Detection

A machine learning system for detecting fraudulent credit card transactions on a **severely imbalanced dataset** (fraud rate < 0.2%) — optimised for real-world payment risk scenarios where missing fraud is costlier than a false alarm.

## What it does

* Handles extreme class imbalance using SMOTE oversampling
* Trains and compares Random Forest and XGBoost models
* Tunes decision thresholds based on real-world cost asymmetry
* Evaluates using AUC-ROC and Precision-Recall curves (not accuracy)
* Includes SQL queries for transaction-level fraud pattern analysis

## Tech Stack
Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, SQL

## Pipeline Structure
Raw Transactions → EDA → SMOTE Oversampling → Model Training → Threshold Tuning → Evaluation → Report

## Key Results

| Metric | Approach |
|---|---|
| Primary metric | AUC-ROC |
| Optimisation target | Recall (minimise missed fraud) |
| Imbalance handling | SMOTE oversampling |
| Best model | XGBoost / Random Forest ensemble |

## How to Run

```bash
git clone https://github.com/Ayesha037/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
jupyter notebook
```

## Key Learnings

* Accuracy is a misleading metric on imbalanced data — always use AUC-ROC and Precision-Recall
* SMOTE significantly improves sensitivity to the minority (fraud) class
* Threshold tuning based on business cost is critical in real payment systems

## Author
**Mohammad Ayesha Summaiyya** — msumaiya03579@gmail.com
