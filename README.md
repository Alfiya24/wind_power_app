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

---

## Getting Started
To run this project locally:
1. Clone the repository:
```bash
git clone https://github.com/Alfiya24/wind_power_app.git
cd wind_power_app
2. Create a virtual environment:
python -m venv venv
venv\Scripts\activate
3. Install requirements:
pip install -r requirements.txt
4. Set your OpenWeatherMap API key inside wind_power_App.py:
API_KEY = "YOUR_API_KEY"
5. Run the wind_power_App:
streamlit run App/wind_power_App.

---

## Preview
[App Screenshot](assets/screenshot.png)

---

## License
This repository is for educational and portfolio purposes only.
