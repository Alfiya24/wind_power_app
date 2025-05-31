# lstm_prepare_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load data
df = pd.read_csv('data/engineered_data.csv', parse_dates = ['Date/Time'])
df.set_index('Date/Time', inplace = True)

# Select features for LSTM
features = ['power_output', 'wind_speed', 'power_curve']
print("Feature preview:\n", df[features].head())

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# Create Sequence
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size, 1:])       # input features: wind_speed, power_curve
        y.append(data[i+window_size, 0])          # target: power_output
    return np.array(X), np.array(y)

# Define sliding window size (e.g., past 6 time steps to predict next)
WINDOW_SIZE = 6
X, y = create_sequences(scaled_data, WINDOW_SIZE)

print(f"✅ Prepared sequences: X shape = {X.shape}, y shape = {y.shape}")

# Save as .npy files for training
np.save('data/X_lstm.npy', X)
np.save('data/y_lstm.npy', y)

# Save scaler for inverse_transform later
import joblib
joblib.dump(scaler, 'app/lstm_scaler.pkl')

print("✅ Saved X_lstm.npy, y_lstm.npy, and lstm_scaler.pkl")
