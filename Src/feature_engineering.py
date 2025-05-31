# feature_engineering.py
import pandas as pd
# Load the cleaned data
df = pd.read_csv('data/cleaned_scada_data.csv', parse_dates=['Date/Time'])
# Set datetime as index
df.set_index('Date/Time', inplace=True)
# Feature Engineering: Extract time-based features
df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
df['minute'] = df.index.minute
# Save the new engineered dataset
df.to_csv('data/engineered_data.csv')
print("✅ Feature engineered dataset saved as 'engineered_data.csv'")
