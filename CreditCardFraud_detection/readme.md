# 💳 Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced binary classification problem.

The project compares **Logistic Regression as a baseline model** with a **Random Forest classifier**, while using SMOTE, feature engineering, stratified splitting, threshold tuning, and fraud-focused evaluation metrics.

---

## 📌 Project Overview

Credit card fraud detection is a challenging machine learning problem because fraudulent transactions represent only a very small fraction of all transactions. A model can achieve high overall accuracy while still failing to detect a significant number of fraudulent transactions.

This project therefore focuses not only on accuracy, but also on:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix
- Precision–Recall trade-offs
- Decision-threshold tuning

The objective is to build a reproducible pipeline that can identify fraudulent transactions while keeping false positives under control.

---

## 🎯 Objectives

1. Load and inspect the credit card transaction dataset.
2. Analyze the severe class imbalance through exploratory data analysis.
3. Investigate potential outliers without automatically removing suspicious transactions.
4. Engineer useful time- and amount-based features.
5. Scale numerical features appropriately.
6. Perform a stratified train/test split.
7. Apply SMOTE **only to the training data** to avoid test-set contamination.
8. Train and compare:
   - Logistic Regression — baseline model
   - Random Forest — primary tree-based model
9. Evaluate both models using fraud-appropriate metrics.
10. Tune the classification threshold using the Precision–Recall–F1 trade-off.
11. Identify the stronger model based on the project's evaluation criteria.

---

## 📊 Dataset

**Dataset:** Credit Card Fraud Detection

**Source:** Kaggle — Machine Learning Group / ULB

The dataset contains anonymized credit card transactions with:

- `Time` — elapsed time associated with the transaction
- `Amount` — transaction amount
- `V1`–`V28` — anonymized PCA-transformed numerical features
- `Class` — target variable

### Target

| Class | Meaning |
|------:|---------|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

The dataset is highly imbalanced, with fraudulent transactions forming a very small percentage of the total records.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🔬 Methodology

The project follows this pipeline:

```text
Raw Dataset
     │
     ▼
Data Loading & Validation
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Outlier Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Feature Scaling
     │
     ▼
Stratified Train/Test Split
     │
     ├──────────────► Test Set
     │                  (unchanged)
     ▼
SMOTE on Training Set
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     │
     └── Random Forest
     │
     ▼
Model Evaluation
     │
     ▼
Threshold Tuning
     │
     ▼
Final Model Selection
```

---

## 🧹 Data Preprocessing

### 1. Data Validation

The notebook checks:

- Dataset shape
- Data types
- Missing values
- Initial records
- Class distribution

### 2. Outlier Analysis

Isolation Forest is used to investigate potential anomalies using `Amount` and `Time`.

An important design decision is made:

> Detected outliers are **not automatically deleted**, because unusual transactions may themselves contain useful fraud signals.

An IQR-based analysis is also performed for the `Amount` feature.

### 3. Feature Engineering

The following features are created:

| Feature | Description |
|---|---|
| `Time_Hour` | Hour extracted from transaction time |
| `Time_Day` | Day derived from transaction time |
| `Amount_log` | Log-transformed transaction amount |
| `Amount_scaled` | Robust-scaled transaction amount |
| `Amount_log_scaled` | Robust-scaled log-transformed amount |
| `Time_scaled` | Standard-scaled transaction time |

The final feature set uses:

- `V1`–`V28`
- engineered amount features
- engineered time features

---

## ⚖️ Handling Class Imbalance

Class imbalance is one of the most important challenges in this project.

### SMOTE

SMOTE (Synthetic Minority Oversampling Technique) is applied **only to the training set**.

```text
Before Split
     │
     ▼
Stratified Train/Test Split
     │
     ├── Test → unchanged
     │
     └── Training → SMOTE
```

This is important because applying SMOTE before the train/test split could introduce synthetic information related to the training data into the test set and produce misleading evaluation results.

The test set remains representative of the original class distribution.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression — Baseline

Logistic Regression is used as the baseline classifier.

Configuration includes:

- `max_iter = 1000`
- `class_weight = "balanced"`
- `random_state = 42`

It provides a simple and interpretable reference point for evaluating the more complex Random Forest model.

### 2. Random Forest — Primary Model

Random Forest is used as the main tree-based classification model.

Current configuration:

```text
n_estimators = 100
max_depth = 15
min_samples_split = 20
min_samples_leaf = 10
class_weight = "balanced"
random_state = 42
```

Random Forest can model nonlinear relationships and interactions between features, making it useful for complex tabular transaction data.

---

## 📈 Evaluation Metrics

Accuracy alone is not sufficient for this project.

### Accuracy

The percentage of all transactions classified correctly.

### Precision

Of all transactions predicted as fraud, how many were actually fraudulent?

High precision helps reduce unnecessary investigation of legitimate customers.

### Recall

Of all actual fraudulent transactions, how many were successfully detected?

For fraud detection, recall is especially important because a false negative means that a fraudulent transaction was missed.

### F1-Score

The harmonic mean of precision and recall.

It provides a balanced measure when both false positives and false negatives matter.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across different thresholds.

### PR-AUC

Measures performance across the precision–recall trade-off.

For a highly imbalanced fraud dataset, PR-AUC is particularly useful because it focuses on performance involving the minority fraud class.

---

## 🧮 Confusion Matrix

The project analyzes four outcomes:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| **Actual Legitimate** | True Negative (TN) | False Positive (FP) |
| **Actual Fraud** | False Negative (FN) | True Positive (TP) |

### Business interpretation

- **False Positive:** A legitimate customer is incorrectly flagged as fraudulent.
- **False Negative:** A fraudulent transaction is incorrectly classified as legitimate.

In a real financial system, false negatives can be particularly costly because they represent missed fraud.

---

## 🎚️ Threshold Tuning

Most binary classifiers use a default probability threshold of `0.50`.

This project does not assume that `0.50` is automatically optimal.

Different thresholds are tested:

```text
0.10 → 0.15 → 0.20 → ... → 0.95
```

For each threshold, the following are calculated:

- Accuracy
- Precision
- Recall
- F1-Score

The threshold producing the highest F1-Score is selected as the project's optimal threshold.

This allows the model to balance fraud detection and false-alarm rates more effectively than relying blindly on the default threshold.

---

## 📊 Visualizations

The notebook includes visual analysis for:

- Class distribution
- Transaction amount distributions
- Transaction time distributions
- Amount boxplots
- Isolation Forest outlier analysis
- ROC curves
- Precision–Recall curves
- Precision vs Recall comparison
- Confusion matrix
- Threshold tuning

---

## 📁 Project Structure

Recommended repository structure:

```text
CreditCardFraud_detection/
│
├── hp1.ipynb
├── README.md
├── creditcard.csv                 # Dataset - do not commit if repository policy excludes it
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── scalers/
│   ├── robust_scaler.pkl
│   └── time_scaler.pkl
│
├── artifacts/
│   └── feature_names.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── threshold_tuning.csv
│
└── plots/
    ├── eda/
    ├── model_comparison/
    └── threshold_tuning/
```

> The exact generated filenames may vary depending on the final model-saving cells used in the notebook.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** — data manipulation
- **NumPy** — numerical computation
- **Matplotlib** — visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — preprocessing, models and evaluation
- **Imbalanced-learn** — SMOTE
- **Joblib / Pickle** — model and artifact serialization
- **Jupyter Notebook / Google Colab** — development environment

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/harshjpatil007/Machine_learning_Projects.git
cd Machine_learning_Projects/CreditCardFraud_detection
```

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

---

## ▶️ Running the Project

1. Download `creditcard.csv` from Kaggle.
2. Place the dataset in the project directory.
3. Open:

```text
hp1.ipynb
```

4. Run the notebook from top to bottom.

The notebook creates directories for:

```text
results/
models/
scalers/
artifacts/
plots/
```

and stores generated outputs in the appropriate locations.

---

## 🔐 Model Inference

The trained model can be used to classify a new transaction after applying the **same feature engineering and scaling steps used during training**.

A prediction consists of:

```text
Transaction
     │
     ▼
Feature Engineering
     │
     ▼
Scaling
     │
     ▼
Trained Model
     │
     ▼
Fraud Probability
     │
     ▼
Decision Threshold
     │
     ├── Legitimate
     └── Fraudulent
```

The probability threshold can be adjusted according to the operational requirements of a fraud-detection system.

---

## 📌 Important Design Decisions

### Why not rely only on accuracy?

Because the dataset is severely imbalanced. A model could classify almost every transaction as legitimate and still obtain a deceptively high accuracy while detecting very little fraud.

### Why use SMOTE?

SMOTE provides synthetic minority-class training examples so that the models receive more information about fraudulent transactions during training.

### Why keep the test set unchanged?

The test set should represent the real-world distribution of transactions. This makes evaluation more realistic.

### Why use Logistic Regression?

It provides a simple baseline against which the more flexible Random Forest model can be compared.

### Why use Random Forest?

Random Forest can capture nonlinear relationships and feature interactions that a linear model may not capture.

### Why tune the threshold?

The default `0.50` decision threshold is not necessarily optimal for fraud detection. Lower or higher thresholds can change the balance between missed fraud and false alarms.

---

## ⚠️ Limitations

This project is an educational machine learning implementation and should not be considered a production-ready banking fraud detection system.

Potential limitations include:

- Anonymized PCA features make business interpretation difficult.
- Historical transaction data may not represent future fraud patterns.
- Fraud strategies evolve over time.
- SMOTE-generated examples may not perfectly represent real fraudulent transactions.
- A production system would require continuous monitoring and retraining.
- Additional operational and customer-level features could improve fraud detection.
- Threshold selection should ultimately consider the financial cost of false positives and false negatives.

---

## 🚀 Future Improvements

Possible extensions include:

- Hyperparameter optimization using GridSearchCV or RandomizedSearchCV
- Stratified cross-validation
- Cost-sensitive learning
- Advanced ensemble models
- XGBoost / LightGBM comparison
- Probability calibration
- Cost-based threshold optimization
- Explainable AI using SHAP
- Real-time fraud scoring API
- Model monitoring and drift detection
- Automated retraining pipeline
- Deployment using FastAPI or Streamlit

---

## 📚 References

1. Kaggle — Credit Card Fraud Detection Dataset  
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Scikit-learn Documentation  
   https://scikit-learn.org/

3. imbalanced-learn Documentation  
   https://imbalanced-learn.org/

4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.  
   *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 2002.

---

## 👨‍💻 Author

**Harsh Patil**

Computer Engineering Student

---

## 📄 Project Status

**Status:** Machine Learning Pipeline Completed / Under Development

The project currently focuses on a reproducible fraud-detection workflow using Logistic Regression and Random Forest, with imbalance handling and threshold optimization.
