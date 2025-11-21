import streamlit as st
import pickle
import pandas as pd

# ==== LOAD MODEL ====
with open("model.pkl", "rb") as f:
    model = pickle.load(f)   # hanya model saja

st.title("Prediksi Churn Pelanggan")

# ==== INPUT DATA ====
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("SeniorCitizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure", min_value=0, max_value=100)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

# ==== ENCODE MANUAL ====
def encode(val):
    return 1 if val in ["Yes", "Male", "Fiber optic", "One year", "Two year"] else 0

input_data = pd.DataFrame({
    "gender": [encode(gender)],
    "SeniorCitizen": [senior],
    "Partner": [encode(partner)],
    "Dependents": [encode(dependents)],
    "tenure": [tenure],
    "PhoneService": [encode(phoneservice)],
    "InternetService": [encode(internet)],
    "Contract": [encode(contract)],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})

# ==== PREDIKSI ====
if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    st.success(f"Hasil Prediksi: {'Churn' if pred == 1 else 'Tidak Churn'}")
