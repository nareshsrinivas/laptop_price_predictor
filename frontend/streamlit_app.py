import streamlit as st
import httpx

st.title("Laptop Price Predictor")

brand = st.selectbox("Brand", ["Dell", "HP", "Lenovo", "Asus", "Acer"])
ram = st.number_input("RAM (GB)", min_value=2, max_value=128, value=8)
storage = st.number_input("Storage (GB)", min_value=128, max_value=4000, value=512)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)
cpu_freq_ghz = st.number_input("CPU Frequency (GHz)", min_value=1.0, max_value=5.0, value=2.5)

if st.button("Predict Price"):
    features = {
        "brand": brand,
        "ram": ram,
        "storage": storage,
        "weight": weight,
        "cpu_freq_ghz": cpu_freq_ghz
    }

    try:
        response = httpx.post("http://127.0.0.1:8000/api/predict", json=features)
        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"Predicted Price: ${price}")
        else:
            st.error(f"Prediction failed: {response.text}")
    except Exception as e:
        st.error(f"Error: {e}")
        
