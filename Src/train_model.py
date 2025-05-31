# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load engineered dataset
df = pd.read_csv('data/engineered_data.csv')

# Step 2: Define features and target
X = df[['wind_speed', 'power_curve', 'hour', 'day', 'month', 'day_of_week', 'is_weekend', 'minute']]
y = df['power_output']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("RMSE:", round(mean_squared_error(y_test, y_pred),))
print("R² Score:", round(r2_score(y_test, y_pred), 4))

# Step 6: Save the model
joblib.dump(model, 'app/model.pkl')
print("✅ Model saved to app/model.pkl")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test[:100], y_pred[:100], alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Power Output")
plt.ylabel("Predicted Power Output")
plt.title("Actual vs Predicted Power Output")
plt.grid(True)
plt.show()

