# 🌬️ Wind Power Forecasting using LSTM & Real-Time Weather Data

This project predicts wind turbine power output using an LSTM model trained on SCADA data, and supports real-time inference using OpenWeatherMap weather API.

---

## Features
- LSTM model trained on time-series SCADA wind turbine data
- Real-time wind speed fetched from OpenWeatherMap API
- Scaled and inverse-transformed power predictions in kW
- Live Streamlit interface for one-click predictions
- Modular scripts for data cleaning, preprocessing, and model training

---

## Project Structure
App/
├── lstm_model.keras # Trained LSTM model
├── lstm_scaler.pkl # Scaler for inverse-transform
├── streamlit_app.py # Streamlit app entry point

Data/
├── scada_data.csv # Raw wind turbine data
├── engineered_data.csv # Cleaned + transformed dataset
├── X_lstm.npy / y_lstm.npy # Prepared time-series sequences

Src/
├── clean_data.py # SCADA cleaning
├── feature_engineering.py # Date/time transformation
├── lstm_prepare_data.py # LSTM input preparation
├── lstm_train_model.py # LSTM training & saving
├── real_time_prediction.py # Live OpenWeatherMap integration

---

## Model Information
- Type: LSTM (Long Short-Term Memory)
- Input: Past 6 time steps of wind speed and theoretical power curve
- Output: Predicted power output (kW)
- Loss: MSE
- Evaluation Metrics: RMSE, MAE, R²

---

## API Reference
OpenWeatherMap API for real-time wind speed data


