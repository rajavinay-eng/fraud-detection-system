# Fraud Detection System

## Live Demo
(https://fraud-detection-system-jhacr2c7dfnnjpd2qvlumn.streamlit.app)

---

End-to-end fraud detection system built on 6.3 million real transactions.

## Results
- ROC-AUC: 0.9946
- Recall: 1.00 — caught every fraud case
- Fraud rate: 0.13% — extreme class imbalance handled with SMOTE

## Tech Stack
Python • XGBoost • SMOTE • SHAP • scikit-learn • joblib • Streamlit

## What It Does
- Detects fraudulent transactions in real-time
- SHAP explainability shows why each transaction was flagged
- Live Streamlit app with instant predictions

## Key Steps
1. Loaded 6.3M real transactions
2. Removed data leakage columns
3. Applied SMOTE to balance classes
4. Trained XGBoost model
5. Added SHAP explanations
6. Deployed Streamlit app

## How To Run
pip install -r requirements.txt
streamlit run app.py
