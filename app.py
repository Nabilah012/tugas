import streamlit as st
import pickle
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model, le = pickle.load(f)

st.title("Prediksi Kategori Mahasiswa")

nilai = st.number_input("Nilai", 0, 100)
penghasilan = st.number_input("Penghasilan Orang Tua", 0)
kehadiran = st.number_input("Persentase Kehadiran", 0, 100)

btn = st.button("Prediksi")

if btn:
    features = pd.DataFrame([[nilai, penghasilan, kehadiran]],
                            columns=["Nilai", "Penghasilan", "Kehadiran"])
    pred = model.predict(features)
    hasil = le.inverse_transform(pred)[0]
    
    st.success(f"Hasil Prediksi: {hasil}")
