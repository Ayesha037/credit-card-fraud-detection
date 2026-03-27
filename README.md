#  Credit Card Fraud Detection

A machine learning system for detecting fraudulent credit card transactions on a severely imbalanced dataset (fraud rate < 0.2%), optimized for real-world payment risk scenarios.

---

##  Problem Statement

Credit card fraud detection is a classic imbalanced classification problem — fraudulent transactions make up less than 0.2% of all data. Standard accuracy metrics are misleading here; the business priority is **minimizing false negatives** (missed fraud), even at the cost of more false positives.

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Models | Random Forest, XGBoost |
| Imbalance Handling | SMOTE (Synthetic Minority Oversampling) |
| Evaluation | AUC-ROC, Precision-Recall, Confusion Matrix |
| Libraries | Scikit-learn, Pandas, NumPy, Matplotlib |

---

##  Key Features

- **SMOTE Oversampling** — Synthetically generates minority class samples to tackle the severe class imbalance
- **Ensemble Methods** — Random Forest and XGBoost trained and compared for best fraud recall
- **Threshold Tuning** — Decision threshold calibrated to real-world cost asymmetry of fraud misclassification
- **Recall-Optimized** — Prioritizes recall over accuracy to minimize financial fraud exposure
- **AUC-ROC Evaluation** — Competitive AUC-ROC score achieved through careful model selection and tuning

---

## Results

| Metric | Value |
|---|---|
| Primary Metric | AUC-ROC |
| Optimization Target | Recall (minimize false negatives) |
| Imbalance Handling | SMOTE oversampling |
| Best Model | XGBoost / Random Forest (Ensemble) |

---

##  How to Run

```bash
# Clone the repo
git clone https://github.com/Ayesha037/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook
```

---

##  Project Structure

```
credit-card-fraud-detection/
│
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
├── models/                # Saved model files
├── requirements.txt
└── README.md
```

---

##  Key Learnings

- Accuracy is a misleading metric for imbalanced datasets — always use AUC-ROC and Precision-Recall curves
- SMOTE significantly improves model sensitivity to minority class
- Threshold tuning based on business cost asymmetry is critical in real payment systems

---

##  Author

**Mohammad Ayesha Summaiyya**  
msumaiya03579@gmail.com  
