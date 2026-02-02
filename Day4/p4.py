import sys
import os
import warnings

warnings.filterwarnings("ignore")

# ---------- Step 1: Read CSV filename ----------
filename = input()

file_path = os.path.join(sys.path[0], filename)
if not os.path.exists(file_path):
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

# ---------- Step 2: Load dataset safely ----------
try:
    import pandas as pd
    df = pd.read_csv(file_path)
except Exception:
    print("Error reading CSV file.")
    sys.exit(1)

# ---------- Step 3: Dataset Preview ----------
print("Dataset Preview:")
print(df.head())
print()

# ---------- Step 4: Dataset Information ----------
print("Dataset Information:")
info = df.info()
print(info)
print()

# ---------- Step 5: Missing Value Check ----------
new_df = df.iloc[:, :-3]
new_df.set_index('Date', inplace=True)

print("Missing Value Check:")
# numeric_cols = new_df.select_dtypes(include=['float64', 'int64']).columns
print(new_df.isnull().sum())

# Handle missing values safely (forward fill)
new_df = new_df.dropna()

print("After missing value handling:")
print(new_df.isnull().sum())
print()

# ---------- Step 6: Train-Test Split ----------
if 'Close' not in df.columns:
    print("Required column 'Close' not found.")
    sys.exit(1)

split_index = int(len(new_df) * 0.8)
train = new_df['Close'][:split_index]
test = new_df['Close'][split_index:]

print("Train-Test Split:")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")
print()

# ---------- Step 7: SARIMA Modeling ----------
print("SARIMA Model Summary:")
try:
    from pmdarima import auto_arima

    model = auto_arima(
        train,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    print(model.summary())

except Exception:
    print("pmdarima not available. SARIMA modeling skipped.")

