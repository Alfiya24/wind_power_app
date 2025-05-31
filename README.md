# 🌬️ Wind Power Forecasting using LSTM & Real-Time Weather Data

This project predicts wind turbine power output using an LSTM model trained on SCADA data, and supports real-time inference using OpenWeatherMap weather API.

---

## 🚀 Features

- 📊 LSTM-based prediction using time-series weather data
- 🔗 Real-time wind speed fetched from OpenWeatherMap
- 📈 Streamlit app for live power forecasting
- ✅ Scaled and inverse-transformed power predictions (in kW)

---

## 🗂️ Project Structure
App/
├── lstm_model.keras # Trained LSTM model
├── lstm_scaler.pkl # Scaler for inverse-transform
├── streamlit_app.py # Final Streamlit app
Data/
├── engineered_data.csv # Final dataset
├── X_lstm.npy / y_lstm.npy
Src/
├── clean_data.py # Data cleaning
├── feature_engineering.py # Feature extraction
├── lstm_prepare_data.py # Sequence generation
├── lstm_train_model.py # LSTM training

