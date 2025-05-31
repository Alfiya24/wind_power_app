import streamlit as st
import requests
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model("app/lstm_model.keras")
scaler = joblib.load("app/lstm_scaler.pkl")

# OpenWeatherMap API setup
API_KEY = "f3abdb5e596d35dafa2e962facd872f8"
CITY = "Dubai"

def get_wind_speed():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    return response["wind"]["speed"]

def make_prediction(wind_speed):
    power_curve = 0.9
    input_data = np.array([[wind_speed, power_curve]])
    scaled_input = scaler.transform(np.array([[0] + list(input_data[0])]))[:, 1:]
    lstm_input = np.repeat(scaled_input[np.newaxis, :, :], 6, axis=1)

    scaled_pred = model.predict(lstm_input).flatten()[0]

    # Inverse transform to get kW value
    dummy_row = np.zeros((1, 3))  # [power_output, wind_speed, power_curve]
    dummy_row[0, 0] = scaled_pred
    inv_pred = scaler.inverse_transform(dummy_row)[0, 0]
    
    return scaled_pred, inv_pred

# Streamlit UI
st.set_page_config(page_title="Wind Power Prediction", layout="centered")
st.title("🌬️ Real-Time Wind Power Prediction")

if st.button("Predict Now"):
    try:
        wind_speed = get_wind_speed()
        st.write(f"✅ Current Wind Speed in {CITY}: **{wind_speed} m/s**")

        scaled_pred, inv_pred = make_prediction(wind_speed)
        st.success(f"⚡ Predicted Power Output: {inv_pred:.2f} kW")
        st.caption(f"(Scaled value: {scaled_pred:.4f})")
    except Exception as e:
        st.error(f"❌ Error fetching prediction: {e}")

st.markdown("---")
st.caption("Model: LSTM | Source: OpenWeatherMap | Location: Dubai")

