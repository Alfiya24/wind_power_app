# real_time_prediction.py

import requests
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import time
import pandas as pd


# 1. Load Trained Model and Scaler
model = load_model('app/lstm_model.keras')
scaler = joblib.load('app/lstm_scaler.pkl')

# 2. Constants
API_KEY = 'f3abdb5e596d35dafa2e962facd872f8'  # Replace with your actual key
LAT = '25.276987'     # Dubai latitude
LON = '55.296249'     # Dubai longitude
WINDOW_SIZE = 6
POWER_CURVE = 0.8     # Assumed constant power curve (or use another logic)

# 3. Fetch Real-Time Wind Speed
def get_real_time_wind_speed():
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    data = response.json()
    wind_speed = data['wind']['speed']
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Wind speed fetched: {wind_speed} m/s")
    return wind_speed

# 4. Prepare input sequence
def prepare_sequence(recent_window):
    # Convert NumPy array to DataFrame with expected feature names
    df_input = pd.DataFrame(recent_window, columns=['power_output', 'wind_speed', 'power_curve'])
    scaled_seq = scaler.transform(df_input)
    X_input = scaled_seq[:, 1:]  # use only wind_speed and power_curve
    return X_input.reshape(1, WINDOW_SIZE, 2)

# 5. Main Logic
def main():
    # Initialize window with dummy past values
    recent_window = [[0, 6.0, POWER_CURVE]] * WINDOW_SIZE  # [power_output, wind_speed, power_curve]

    while True:
        try:
            # Get new wind speed
            wind_speed = get_real_time_wind_speed()
            new_entry = [0, wind_speed, POWER_CURVE]  # power_output dummy (not needed)

            # Slide window
            recent_window.pop(0)
            recent_window.append(new_entry)

            # Prepare input & predict
            X_seq = prepare_sequence(np.array(recent_window))
            predicted_power = model.predict(X_seq)[0][0]

            # Inverse scale if needed
            print(f"⚡ Predicted Power Output: {round(predicted_power, 3)} (scaled value)")

            # Wait 1 minute (or adjust)
            time.sleep(60)
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            break

if __name__ == "__main__":
    main()
