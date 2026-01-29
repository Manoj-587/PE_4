import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os, sys


file_name = input().strip()
df = pd.read_csv(os.path.join(sys.path[0], file_name))

df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['DATE'])
df.set_index('DATE', inplace=True)

print("First 5 records of dataset:")
print(df[['Consumption', 'Festivals/Special_events']].head())
print()

df['Consumption'] = df['Consumption'].fillna(df['Consumption'].mean())

Q1 = df['Consumption'].quantile(0.25)
Q3 = df['Consumption'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df['Consumption'] = np.where(
    df['Consumption'] < lower_bound,
    df['Consumption'].mean(),
    df['Consumption']
)

df['Consumption'] = np.where(
    df['Consumption'] > upper_bound,
    df['Consumption'].mean(),
    df['Consumption']
)

print("Data preprocessing completed.\n")

additive = seasonal_decompose(df['Consumption'], model='additive', period=12)

print("Additive Model Components (First 5 Values)")
print("Trend:")
print(additive.trend.dropna().head())
print()

print("Seasonality:")
print(additive.seasonal.head())
print()

print("Residuals:")
print(additive.resid.dropna().head())
print()

multiplicative = seasonal_decompose(df['Consumption'], model='multiplicative', period=12)

print("Multiplicative Model Components (First 5 Values)")
print("Trend:")
print(multiplicative.trend.dropna().head())
print()

print("Seasonality:")
print(multiplicative.seasonal.head())
print()

print("Residuals:")
print(multiplicative.resid.dropna().head())
print()

print("Model Comparison Conclusion:")
print("If seasonal values are constant → Additive model fits better.")
print("If seasonal values change proportionally with trend → Multiplicative model fits better.")
