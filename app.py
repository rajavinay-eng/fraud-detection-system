import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load saved model and scaler
model = joblib.load(r'C:/Users/RAJA VINAY KUMAR/Downloads/job preparation/fraud_model.pkl')
scaler = joblib.load(r'C:/Users/RAJA VINAY KUMAR/Downloads/job preparation/fraud_scaler.pkl')

st.title("🏦 UBS Fraud Detection System")
st.write("Enter transaction details to check if it is fraudulent.")

# Input fields
step = st.number_input("Step (time)", min_value=1, max_value=744, value=1)
type_map = {'PAYMENT': 3, 'TRANSFER': 4, 'CASH_OUT': 1, 'DEBIT': 0, 'CASH_IN': 2}
type_input = st.selectbox("Transaction Type", list(type_map.keys()))
type_encoded = type_map[type_input]
amount = st.number_input("Amount (CHF)", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Sender Balance Before", min_value=0.0, value=5000.0)
newbalanceOrig = st.number_input("Sender Balance After", min_value=0.0, value=4000.0)
oldbalanceDest = st.number_input("Receiver Balance Before", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Receiver Balance After", min_value=0.0, value=0.0)

# Predict button
if st.button("Check Transaction"):
    input_data = np.array([[step, type_encoded, amount,
                            oldbalanceOrg, newbalanceOrig,
                            oldbalanceDest, newbalanceDest]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED — Confidence: {probability:.1%}")
    else:
        st.success(f"✅ GENUINE TRANSACTION — Fraud probability: {probability:.1%}")