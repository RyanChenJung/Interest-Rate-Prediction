# 📈 Loan Interest Rate Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=flat-square)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green?style=flat-square)
![RMSE](https://img.shields.io/badge/Test%20RMSE-1.37-brightgreen?style=flat-square)

## 📌 Project Overview
This project implements an end-to-end machine learning pipeline to predict the **Interest Rate ($X1$)** assigned to loans based on borrower financial data (e.g., credit grade, annual income, loan purpose, and debt-to-income ratio).

The goal was to build a regression model that minimizes the **Root Mean Squared Error (RMSE)** while identifying key risk factors. The final model achieved a Test RMSE of **1.37**, representing a significant improvement over the linear baseline.

## 🚀 Key Results
We compared linear methods against ensemble tree methods. The non-linear approach proved superior in capturing complex credit risk patterns.

| Model | Technique | Test RMSE | Performance |
| :--- | :--- | :--- | :--- |
| **Baseline** | Lasso Regression (L1) | 1.7832 | Good interpretability, but underfitted. |
| **Final Model** | **Random Forest Regressor** | **1.3717** | **~23% Improvement** in accuracy. |

## 🛠️ Methodology

### 1. Data Cleaning & Preprocessing
* **Date Parsing:** Corrected "Century Rollover" errors in credit history dates (e.g., distinguishing 19xx from 20xx) and fixed mixed date formats.
* **String Cleaning:** Removed artifacts (`%`, `$`) from numerical columns to ensure proper data typing.
* **Imputation:** Applied domain-specific imputation for missing values (e.g., Median for Income, 'Unknown' for categorical features).

### 2. Feature Engineering
* **Target Encoding:** Applied Smoothed Target Encoding to the `State` ($X20$) variable to capture geographic risk signals without high dimensionality.
* **Ratio Features:** Created new financial ratios (e.g., `Funded Ratio`, `Credit History Length`) to expose latent signals.
* **Log Transformation:** Applied log transformation to skewed features (e.g., Annual Income) to normalize distributions.

### 3. Feature Selection Strategy
* **Lasso (L1):** Used initially to identify and remove strictly redundant linear features.
* **Recursive Feature Elimination (RFE):** Used with a Random Forest estimator to analyze non-linear feature importance.
* **Final Decision:** Adopted a **"Greedy Strategy"** (utilizing all 39 features) for the final Random Forest model to capture all potential interactions and maximize predictive power.

### 4. Model Optimization
* **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` with 5-fold Cross-Validation on a training subset.
* **Best Parameters:** `n_estimators=300`, `max_depth=25`, `min_samples_leaf=2`.
* **Diagnostics:** Validated model stability using Learning Curves to ensure no severe overfitting occurred.

## 📂 Repository Structure
```
├── data/                   # (Optional) Folder for raw data
├── notebooks/              # Jupyter Notebooks containing the full pipeline
├── results/                # Generated prediction files
│   └── Results_from_Ryan_Chen_Jung.csv  # Final predictions for the holdout set
├── models/                 # Serialized models (.pkl)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```


## 💻 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter Notebook to execute the training pipeline and generate predictions.
