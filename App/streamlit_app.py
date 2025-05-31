# streamlit_app.py

import streamlit as st
import joblib
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

st.title("💨 Wind Power Forecasting App")
st.markdown("Enter weather and time details to predict power output (in kW).")

# User inputs
wind_speed = st.slider("Wind Speed (m/s)", 0.0, 25.0, 5.0)
power_curve = st.slider("Theoretical Power Curve (kWh)", 0.0, 4000.0, 500.0)
hour = st.slider("Hour of Day", 0, 23, 12)
day = st.slider("Day of Month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
is_weekend = st.selectbox("Is it a Weekend?", ["No", "Yes"])
minute = st.slider("Minute", 0, 59, 0)

# Convert weekend to binary
is_weekend = 1 if is_weekend == "Yes" else 0

# Prepare input
features = np.array([[wind_speed, power_curve, hour, day, month, day_of_week, is_weekend, minute]])

# Predict
if st.button("Predict Power Output"):
    prediction = model.predict(features)[0]
    st.success(f"⚡ Predicted Power Output: **{round(prediction, 2)} kW**")
