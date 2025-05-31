# lstm_train_model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load prepared data
X = np.load('data/X_lstm.npy')   # shape: (samples, time_steps, features)
y = np.load('data/y_lstm.npy')   # shape: (samples,)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
model.add(Dropout(0.2)),
model.add(LSTM(32)),
model.add(Dropout(0.2)),
model.add(Dense(1))  # Predicting one value: power_output

# Compile model with tuned learning
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Early stopping with more patience
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_pred = model.predict(X_test).flatten()
print("y_pred shape:", y_pred.shape)
print("y_test shape:", y_test.shape)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {round(rmse, 2)}")
print(f"✅ MAE: {round(mae, 2)}")
print(f"✅ R² Score: {round(r2, 4)}")

# Save model
model.save('app/lstm_model.keras')
print("✅ LSTM model saved to app/lstm_model.keras")
