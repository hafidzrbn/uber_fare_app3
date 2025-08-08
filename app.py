# ============================================================
# 🚖 UBER FARE PREDICTION APP
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
from datetime import datetime
import os

# ============================================================
# 1️⃣ Load Best Model (dari root folder)
# ============================================================
@st.cache_resource
def load_model():
    """
    Memuat pipeline model terbaik dari file joblib.
    """
    model_path = "random_forest_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ File model '{model_path}' tidak ditemukan. "
                 "Pastikan file sudah di-upload ke root repo GitHub.")
        st.stop()
    return joblib.load(model_path)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ============================================================
# 2️⃣ Fungsi Feature Engineering
# ============================================================
# Replikasi fungsi Haversine dari notebook
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = R * c

    return distance_km

# ============================================================
# 3️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

st.title("🚖 Uber Fare Prediction App")
st.markdown("Masukkan detail perjalanan untuk memprediksi tarif Uber.")

# ============================================================
# 4️⃣ Input Form
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 Trip Details Input")

    col1, col2 = st.columns(2)
    
    with col1:
        pickup_year = st.slider("Pickup Year", 2009, 2015, 2012)
        pickup_month = st.slider("Pickup Month", 1, 12, 6)
        pickup_day = st.slider("Pickup Day", 1, 31, 15)
        pickup_hour = st.slider("Pickup Hour", 0, 23, 12)
    
    with col2:
        pickup_latitude = st.number_input("Pickup Latitude", value=40.738354, format="%.6f")
        pickup_longitude = st.number_input("Pickup Longitude", value=-73.999817, format="%.6f")
        dropoff_latitude = st.number_input("Dropoff Latitude", value=40.723217, format="%.6f")
        dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.999512, format="%.6f")

    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)

    submitted = st.form_submit_button("🔮 Predict Fare")

    if submitted:
        try:
            # Replicate feature engineering pipeline from notebook
            
            # Create a base dataframe from raw inputs
            input_data = pd.DataFrame([{
                'pickup_longitude': pickup_longitude,
                'pickup_latitude': pickup_latitude,
                'dropoff_longitude': dropoff_longitude,
                'dropoff_latitude': dropoff_latitude,
                'passenger_count': passenger_count,
                'pickup_datetime': pd.to_datetime(f'{pickup_year}-{pickup_month}-{pickup_day} {pickup_hour}:00:00')
            }])
            
            # Feature Engineering: Haversine distance
            input_data['trip_distance_km'] = haversine(
                input_data['pickup_latitude'], input_data['pickup_longitude'],
                input_data['dropoff_latitude'], input_data['dropoff_longitude']
            )

            # Feature Engineering: Temporal features
            input_data['year'] = input_data['pickup_datetime'].dt.year
            input_data['month'] = input_data['pickup_datetime'].dt.month
            input_data['day'] = input_data['pickup_datetime'].dt.day
            input_data['dayofweek'] = input_data['pickup_datetime'].dt.dayofweek
            input_data['hour'] = input_data['pickup_datetime'].dt.hour
            
            # Drop unnecessary columns to match the trained model's features
            # The model was trained with specific features, so we must match that.
            # Based on the error message, 'Unnamed: 0' and 'key' columns were
            # also dropped during training, but they are not created here.
            # We need to drop the original 'pickup_datetime' column.
            input_data = input_data.drop(columns=['pickup_datetime'])

            # Make prediction
            prediction = model.predict(input_data)
            
            # Tampilkan hasil
            st.subheader("✅ Prediction Successful!")
            st.success(f"Predicted Uber fare: ${prediction[0]:,.2f}")
            
            st.markdown("---")
            st.subheader("🔍 Input Data Sent to Model")
            st.dataframe(input_data)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")