import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# -----------------------------
# Load labeled CSV
# -----------------------------
df = pd.read_csv("cpu_usage_labeled.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
df["rolling_mean_5"] = df["cpu_usage"].rolling(5).mean()
df["rolling_std_5"] = df["cpu_usage"].rolling(5).std()
df["rolling_max_5"] = df["cpu_usage"].rolling(5).max()
df["delta_usage"] = df["cpu_usage"].diff()

mean = df["cpu_usage"].mean()
std = df["cpu_usage"].std()
df["z_score"] = (df["cpu_usage"] - mean) / std

df.dropna(inplace=True)

# -----------------------------
# Train / Test Split
# -----------------------------
X = df[
    ["cpu_usage", "rolling_mean_5", "rolling_std_5",
     "rolling_max_5", "delta_usage", "z_score"]
]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Random Forest Model
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight={0: 1, 1: 3},
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(rf, "cpu_anomaly_rf.pkl")

print("Model trained and saved as cpu_anomaly_rf.pkl")
