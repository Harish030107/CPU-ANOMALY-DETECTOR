import pandas as pd
import numpy as np
import psutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Parameters
# -----------------------------
COLLECT_SECONDS = 120
WINDOW_SIZE = 10
TOP_PERCENT = 0.9  # Keep values that cover 90% of frequency

data = []

# Collect CPU usage
for _ in range(COLLECT_SECONDS):
    cpu = psutil.cpu_percent(interval=1)
    data.append({"Timestamp": datetime.now(), "cpu_usage": cpu})

df = pd.DataFrame(data)

# Rolling features
df["rolling_mean"] = df["cpu_usage"].rolling(WINDOW_SIZE).mean()
df["rolling_std"]  = df["cpu_usage"].rolling(WINDOW_SIZE).std()
df["rolling_min"]  = df["cpu_usage"].rolling(WINDOW_SIZE).min()
df["rolling_max"]  = df["cpu_usage"].rolling(WINDOW_SIZE).max()
df["delta"]        = df["cpu_usage"].diff()
df.dropna(inplace=True)

# -----------------------------
# Normal region based on frequent values
# -----------------------------
# Bin CPU usage to create frequency distribution
bins = np.arange(0, df["cpu_usage"].max() + 1, 0.5)  # 0.5% bins
hist, bin_edges = np.histogram(df["cpu_usage"], bins=bins)

# Sort bins by frequency
freq_indices = np.argsort(hist)[::-1]  # descending order
cumulative = 0
total = hist.sum()
selected_bins = []

# Select bins covering TOP_PERCENT of total occurrences
for idx in freq_indices:
    selected_bins.append(idx)
    cumulative += hist[idx]
    if cumulative / total >= TOP_PERCENT:
        break

# Determine the normal region
normal_min = bin_edges[min(selected_bins)]
normal_max = bin_edges[max(selected_bins)+1]

print(f"[INFO] Normal region based on frequent values: {normal_min:.2f} - {normal_max:.2f}")

# Label anomalies
df["label"] = np.where(
    (df["cpu_usage"] < normal_min) | (df["cpu_usage"] > normal_max),
    1,
    0
)

# -----------------------------
# Features and ML
# -----------------------------
FEATURES = ["cpu_usage", "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "delta"]
X = df[FEATURES]
y = df["label"]

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X, y)

joblib.dump(rf, "rf_freq_model.pkl")
df.to_csv("cpu_training_data_freq.csv", index=False)

print("[OK] Model trained and saved as cpu_rf_freq_model.pkl")
