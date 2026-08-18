import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model & Scaler
model = joblib.load('credit_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("💳 Credit Scoring & Risk Prediction System")
st.write("Enter applicant details to assess creditworthiness.")

# Form Inputs
income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000)
age = st.slider("Age", 18, 70, 30)
loan = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=100000, value=10000)
credit_hist = st.selectbox("Credit History Clean?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
emp_status = st.selectbox("Employment Status", options=[0, 1, 2], format_func=lambda x: ["Unemployed", "Employed", "Self-Employed"][x])

if st.button("Calculate Credit Score & Risk"):
    features = np.array([[income, age, loan, credit_hist, emp_status]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    proba = model.predict_proba(scaled_features)[0][1]

    st.subheader("Results:")
    if prediction == 0:
        st.success(f"✅ Approved / Low Risk! (Default Probability: {proba*100:.1f}%)")
    else:
        st.error(f"⚠️ High Risk / Rejected (Default Probability: {proba*100:.1f}%)")
