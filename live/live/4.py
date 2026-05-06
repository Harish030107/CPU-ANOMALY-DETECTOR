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
TOP_PERCENT = 0.9  # fraction of most frequent values to include

data = []

# -----------------------------
# Collect CPU usage
# -----------------------------
for _ in range(COLLECT_SECONDS):
    cpu = psutil.cpu_percent(interval=1)
    data.append({"Timestamp": datetime.now(), "cpu_usage": cpu})

df = pd.DataFrame(data)

# -----------------------------
# Rolling features
# -----------------------------
df["rolling_mean"] = df["cpu_usage"].rolling(WINDOW_SIZE).mean()
df["rolling_std"]  = df["cpu_usage"].rolling(WINDOW_SIZE).std()
df["rolling_min"]  = df["cpu_usage"].rolling(WINDOW_SIZE).min()
df["rolling_max"]  = df["cpu_usage"].rolling(WINDOW_SIZE).max()
df["delta"]        = df["cpu_usage"].diff()
df.dropna(inplace=True)

# -----------------------------
# Normal region based on frequent values (both low and high)
# -----------------------------
bins = np.arange(0, df["cpu_usage"].max() + 1, 0.5)
hist, bin_edges = np.histogram(df["cpu_usage"], bins=bins)

# Sort bins by frequency (descending)
freq_indices = np.argsort(hist)[::-1]

# Select bins covering TOP_PERCENT of occurrences
cumulative = 0
total = hist.sum()
selected_bins = []

for idx in freq_indices:
    selected_bins.append(idx)
    cumulative += hist[idx]
    if cumulative / total >= TOP_PERCENT:
        break

# Normal region boundaries
normal_min = bin_edges[min(selected_bins)]
normal_max = bin_edges[max(selected_bins)+1]

print(f"[INFO] Normal region: {normal_min:.2f} - {normal_max:.2f}")

# -----------------------------
# Label anomalies (both low and high spikes)
# -----------------------------
df["label"] = np.where(
    (df["cpu_usage"] < normal_min) | (df["cpu_usage"] > normal_max),
    1,
    0
)

# -----------------------------
# Features for ML
# -----------------------------
FEATURES = ["cpu_usage", "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "delta"]
X = df[FEATURES]
y = df["label"]

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
rf.fit(X, y)

# Save model and training data
joblib.dump(rf, "rf_labeled_model.pkl")
df.to_csv("cpu_training_data_freq_minmax.csv", index=False)

print("[OK] Model trained and saved as cpu_rf_freq_minmax_model.pkl")
