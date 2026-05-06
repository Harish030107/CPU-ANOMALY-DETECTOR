import streamlit as st
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import time
import os 
import subprocess

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Live CPU Anomaly Detection",
    page_icon="🖥️",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------

MODEL_PATH = "rf_labeled_model.pkl"

# -----------------------------
# Check if model exists
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Training model using live CPU data...")

    # Run training script (5.py)
    subprocess.run(["python", "4.py"], check=True)

    st.success("Model trained successfully.")

# -----------------------------
# Load model
# -----------------------------
rf = joblib.load(MODEL_PATH)

# -----------------------------
# Session state
# -----------------------------
if "cpu_data" not in st.session_state:
    st.session_state.cpu_data = []

# -----------------------------
# UI
# -----------------------------
st.title("Live CPU Usage Anomaly Detection")
st.write("Random Forest based real-time monitoring")

# -----------------------------
# Collect live CPU usage
# -----------------------------
cpu = psutil.cpu_percent(interval=1)
now = datetime.now()

st.session_state.cpu_data.append({
    "Timestamp": now,
    "cpu_usage": cpu
})

# Keep recent history
st.session_state.cpu_data = st.session_state.cpu_data[-50:]

df = pd.DataFrame(st.session_state.cpu_data)

# -----------------------------
# Feature Engineering
# -----------------------------
WINDOW_SIZE = 5

df["rolling_mean"] = df["cpu_usage"].rolling(WINDOW_SIZE).mean()
df["rolling_std"]  = df["cpu_usage"].rolling(WINDOW_SIZE).std()
df["rolling_min"]  = df["cpu_usage"].rolling(WINDOW_SIZE).min()
df["rolling_max"]  = df["cpu_usage"].rolling(WINDOW_SIZE).max()
df["delta"]        = df["cpu_usage"].diff()

df.dropna(inplace=True)

FEATURES = [
    "cpu_usage",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "delta"
]

# -----------------------------
# Predict anomalies
# -----------------------------
if len(df) >= WINDOW_SIZE:
    df["anomaly"] = rf.predict(df[FEATURES])
else:
    df["anomaly"] = 0

# -----------------------------
# Plot
# -----------------------------
st.subheader("CPU Usage Over Time")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df["Timestamp"], df["cpu_usage"], label="CPU Usage")

ax.scatter(
    df[df["anomaly"] == 1]["Timestamp"],
    df[df["anomaly"] == 1]["cpu_usage"],
    s=80,
    label="Anomaly"
)

ax.set_xlabel("Time")
ax.set_ylabel("CPU Usage (%)")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# -----------------------------
# Anomaly Table
# -----------------------------
st.subheader("Detected Anomalies")
st.dataframe(df[df["anomaly"] == 1][["Timestamp", "cpu_usage"]])

# -----------------------------
# Auto refresh
# -----------------------------
time.sleep(1)
st.rerun()
