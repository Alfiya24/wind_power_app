# Step 1: Load the dataset
import pandas as pd
# Load raw data
df = pd.read_csv('Data/scada_data.csv')
# Preview dataset
print("Shape:", df.shape)
df.head()
# Step 2: Check for missing values
df.info()
print("\nMissing values per column:\n", df.isnull().sum())
# Converting 'Date/Time'
df['Date/Time'] = pd.to_datetime(df['Date/Time'], format="%d %m %Y %H:%M", dayfirst=True)
print(df.head())
# Rename columns for easier use
df.rename(columns={
    'LV ActivePower (kW)': 'power_output',
    'Wind Speed (m/s)': 'wind_speed',
    'Theoretical_Power_Curve (KWh)': 'power_curve',
    'Wind Direction (°)': 'wind_direction'
}, inplace=True)
# Drop Duplicates or Nulls (if any)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
# Filter any extreme outliers
df = df[(df['power_output'] >= 0) & (df['wind_speed'] >= 0)]
#Compute of correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix) 
#Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
# Droping column
df.drop(columns=['wind_direction'], inplace=True)
print(df.head())
# clean_data.py (Add at the end)
df.to_csv('data/cleaned_scada_data.csv')
print("✅ Cleaned dataset saved as 'cleaned_scada_data.csv'")
