from flask import Flask, render_template, request,jsonify 
from influxdb import InfluxDBClient
from flask_cors import CORS
import pandas as pd 
import joblib
from datetime import datetime, timedelta, timezone
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load Pre-trained Model
# -------------------------------

model = joblib.load("cpu_anomaly_rf.pkl")

# -------------------------------
# InfluxDB Configuration
# -------------------------------
client = InfluxDBClient(
    host="195.201.164.158",  
    port=9086,  
    database="end_user_monitoring"
)

# -------------------------------------------------
# Email Configuration
# -------------------------------------------------
SENDER_EMAIL = "signinsih@gmail.com"
RECEIVER_EMAIL = "s.lokeeshkumar006@gmail.com"
EMAIL_PASSWORD = "ydrjjgslvimnbqqu"

# -------------------------------------------------
# Anomaly Tracking (5-minute persistence)
# -------------------------------------------------
anomaly_tracker = {}
ANOMALY_DURATION = timedelta(minutes=5)

#-----------------------------------------
#graph data
#-----------------------------------------
@app.route("/api/recent", methods=["POST"])
def recent_cpu():
    data = request.json
    host = data.get("host")
    floor = data.get("floor")

    if not host or not floor:
        return jsonify([])

    query = f"""
    SELECT usage_idle
    FROM cpu
    WHERE host = '{host}'
      AND floor_name = '{floor}'
      AND time > now() - 15m
    ORDER BY time ASC
    """

    result = client.query(query)
    points = list(result.get_points())

    if not points:
        return jsonify([])

    df = pd.DataFrame(points)

    # -----------------------------
    # CPU USAGE
    # -----------------------------
    df["cpu_usage"] = 100 - df["usage_idle"]

    # -----------------------------
    # FEATURE ENGINEERING (MATCH TRAINING)
    # -----------------------------
    WINDOW = 5

    df["rolling_mean_5"] = df["cpu_usage"].rolling(WINDOW).mean()
    df["rolling_std_5"]  = df["cpu_usage"].rolling(WINDOW).std()
    df["rolling_max_5"]  = df["cpu_usage"].rolling(WINDOW).max()
    df["delta_usage"]    = df["cpu_usage"].diff()

    df["z_score"] = (
        (df["cpu_usage"] - df["rolling_mean_5"]) / df["rolling_std_5"]
    )

    df.dropna(inplace=True)

    if df.empty:
        return jsonify([])

    # -----------------------------
    # 🔐 EXACT FEATURE ORDER FROM MODEL
    # -----------------------------
    feature_cols = list(model.feature_names_in_)
    X = df[feature_cols]

    # -----------------------------
    # PREDICTION
    # -----------------------------
    df["anomaly"] = model.predict(X)

    # -----------------------------
    # RESPONSE FOR GRAPH
    # -----------------------------
    response = df[["time", "cpu_usage", "anomaly"]].to_dict(
        orient="records"
    )

    return jsonify(response)



# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------
def prepare_features(df):
    df["rolling_mean_5"] = df["cpu_usage"].rolling(5).mean()
    df["rolling_std_5"] = df["cpu_usage"].rolling(5).std()
    df["rolling_max_5"] = df["cpu_usage"].rolling(5).max()
    df["delta_usage"] = df["cpu_usage"].diff()
    df["z_score"] = (
        (df["cpu_usage"] - df["rolling_mean_5"])
        / df["rolling_std_5"]
    )
    return df

# -------------------------------------------------
# Email Sender
# -------------------------------------------------
def send_anomaly_email(host, floor, cpu_usage):
    subject = "CPU Anomaly Alert – Sustained for 5 Minutes"

    body = f"""
CPU Anomaly Detected (Sustained)

Host       : {host}
Floor      : {floor}
CPU Usage  : {cpu_usage:.2f} %
Duration   : > 5 minutes

Recommended Actions:
- Check running processes
- Inspect abnormal services
- Validate workload legitimacy
"""

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.send_message(msg)

#-------------------------------------------------
# API Endpoint for Sustained Anomalies
#-------------------------------------------------
# API Endpoint for Sustained Anomalies
#-------------------------------------------------
@app.route("/api/persistent-anomalies")
def get_sustained_anomalies():
    sustained = []
    now = datetime.utcnow()
    five_min_threshold = timedelta(minutes=5)
    
    # Show only anomalies that have persisted for at least 5 minutes
    for (host, floor), info in anomaly_tracker.items():
        duration = now - info["start_time"]
        duration_minutes = int(duration.total_seconds() / 60)
        
        # Only include anomalies that have persisted for 5 minutes or more
        if duration >= five_min_threshold:
            sustained.append({
                "host": host,
                "floor_name": floor,
                "lab_name": info.get("department", "Unknown"),
                "cpu_usage": info.get("cpu_usage", "N/A"),
                "duration_minutes": duration_minutes
            })

    return jsonify(sustained)

# -------------------------------------------------
# Core Anomaly Detection Logic
'''SELECT LAST(usage_idle) AS usage_idle
FROM cpu
WHERE customer_name='MCE'
GROUP BY floor_name, host
'''
# -------------------------------------------------
from datetime import datetime, timedelta

STALE_THRESHOLD = timedelta(seconds=90)  # 1.5 minutes - balance between freshness and tolerance

def run_anomaly_detection():

    # -----------------------------
    # Queries
    # -----------------------------
    uptime_query = """
    SELECT last("uptime") AS uptime
    FROM "system"
    WHERE customer_name='MCE'
    GROUP BY host, floor_name
    """

    cpu_query = """
    SELECT last("usage_idle") AS usage_idle
    FROM "cpu"
    WHERE customer_name='MCE'
    GROUP BY host, floor_name, department
    """

    cpu_query_fallback = """
    SELECT last("usage_idle") AS usage_idle
    FROM "cpu"
    WHERE customer_name='MCE'
    GROUP BY host, floor_name
    """

    uptime_result = client.query(uptime_query)

    try:
        cpu_result = client.query(cpu_query)
        if not cpu_result:
            cpu_result = client.query(cpu_query_fallback)
    except Exception:
        cpu_result = client.query(cpu_query_fallback)

    rows = []

    # -----------------------------
    # Parse uptime data
    # -----------------------------
    uptime_data = {}
    for (measurement, tags), points in uptime_result.items():
        point = next(points, None)
        if point is None:
            continue

        uptime = point.get("uptime")
        if uptime is None:
            continue

        key = (
            tags.get("host", "UNKNOWN"),
            tags.get("floor_name", "UNKNOWN")
        )
        uptime_data[key] = uptime

    # -----------------------------
    # Parse CPU data (WITH TIME CHECK)
    # -----------------------------
    now = datetime.utcnow()

    for (measurement, tags), points in cpu_result.items():
        point = next(points, None)
        if point is None:
            continue

        key = (
            tags.get("host", "UNKNOWN"),
            tags.get("floor_name", "UNKNOWN")
        )

        # Skip if uptime missing
        if key not in uptime_data:
            continue

        cpu_idle = point.get("usage_idle")
        point_time = point.get("time")

        if cpu_idle is None or point_time is None:
            continue

        # -----------------------------
        # STALE DATA PROTECTION (CRITICAL)
        # -----------------------------
        point_time_dt = datetime.fromisoformat(
            point_time.replace("Z", "+00:00")
        )

        now = datetime.now(timezone.utc)

        if now - point_time_dt > STALE_THRESHOLD:
            continue

        cpu_usage = 100 - cpu_idle
        department = tags.get(
            "department",
            tags.get("floor_name", "Unknown")
        )

        rows.append({
            "host": key[0],
            "floor_name": key[1],
            "department": department,
            "cpu_usage": cpu_usage
        })

    # -----------------------------
    # Build DataFrame
    # -----------------------------
    df = pd.DataFrame(rows)
    if df.empty:
        print("No live CPU data found")
        return []

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    df_feat = prepare_features(df)

    feature_cols = [
        "cpu_usage",
        "rolling_mean_5",
        "rolling_std_5",
        "rolling_max_5",
        "delta_usage",
        "z_score"
    ]

    df_feat.dropna(subset=feature_cols, inplace=True)
    if df_feat.empty:
        return []

    # -----------------------------
    # ML Anomaly Detection
    # -----------------------------
    X = df_feat[feature_cols]
    df_feat["anomaly"] = model.predict(X)

    anomalies = df_feat[df_feat["anomaly"] == 1]

    # -----------------------------
    # Sustained Anomaly Tracking
    # -----------------------------
    current_time = datetime.utcnow()
    current_keys = set(
        zip(anomalies["host"], anomalies["floor_name"])
    )

    # Remove cleared anomalies
    for key in list(anomaly_tracker.keys()):
        if key not in current_keys:
            del anomaly_tracker[key]

    # Track active anomalies
    for _, row in anomalies.iterrows():
        key = (row["host"], row["floor_name"])

        if key not in anomaly_tracker:
            anomaly_tracker[key] = {
                "start_time": current_time,
                "email_sent": False,
                "cpu_usage": row["cpu_usage"],
                "department": row.get("department", "Unknown")
            }
        else:
            info = anomaly_tracker[key]
            info["cpu_usage"] = row["cpu_usage"]

            if (
                not info["email_sent"]
                and current_time - info["start_time"] >= ANOMALY_DURATION
            ):
                send_anomaly_email(
                    host=row["host"],
                    floor=row["floor_name"],
                    cpu_usage=row["cpu_usage"]
                )
                info["email_sent"] = True

    # -----------------------------
    # Output (department → lab_name)
    # -----------------------------
    result_list = anomalies[
        ["host", "floor_name", "department", "cpu_usage"]
    ].to_dict(orient="records")

    for item in result_list:
        item["lab_name"] = item["department"]

    return result_list


# -------------------------------------------------
# API Endpoint (optional dashboard use)
# -------------------------------------------------
@app.route("/api/anomalies")
def api_anomalies():
    anomalies = run_anomaly_detection()
    return jsonify(anomalies)
#--------------------------------------------
#online offline count
#-------------------------------------------
ONLINE_THRESHOLD = timedelta(minutes=5)  # 5 minutes - systems reporting within 5 min are online 

def get_system_online_status():
    """
    Returns:
    {
        "online": int,
        "offline": int,
        "total": int,
        "details": [ {host, floor_name, status, last_seen} ]
    }
    """

    query = """
    SELECT LAST("usage_idle") AS usage_idle
    FROM "cpu"
    WHERE customer_name = 'MCE'
    GROUP BY host, floor_name
    """

    result = client.query(query)

    now = datetime.now(timezone.utc)

    online_count = 0
    offline_count = 0
    details = []

    for (_, tags), points in result.items():
        point = next(points, None)
        if point is None:
            offline_count += 1
            continue

        host = tags.get("host", "UNKNOWN")
        floor = tags.get("floor_name", "UNKNOWN")

        point_time = point.get("time")
        if not point_time:
            offline_count += 1
            details.append({
                "host": host,
                "floor_name": floor,
                "status": "OFFLINE",
                "last_seen": None
            })
            continue

        # Convert InfluxDB time → UTC-aware datetime
        try:
            # Handle both string and datetime objects from InfluxDB
            if isinstance(point_time, str):
                # Parse ISO format timestamp
                if point_time.endswith('Z'):
                    point_time_dt = datetime.fromisoformat(point_time.replace("Z", "+00:00"))
                else:
                    point_time_dt = datetime.fromisoformat(point_time)
                    if point_time_dt.tzinfo is None:
                        point_time_dt = point_time_dt.replace(tzinfo=timezone.utc)
            else:
                # Already a datetime object
                point_time_dt = point_time
                if point_time_dt.tzinfo is None:
                    point_time_dt = point_time_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"Error parsing time for {host}: {e}, time value: {point_time}")
            offline_count += 1
            details.append({
                "host": host,
                "floor_name": floor,
                "status": "OFFLINE",
                "last_seen": str(point_time)
            })
            continue

        time_diff = now - point_time_dt
        
        if time_diff <= ONLINE_THRESHOLD:
            status = "ONLINE"
            online_count += 1
        else:
            status = "OFFLINE"
            offline_count += 1

        details.append({
            "host": host,
            "floor_name": floor,
            "status": status,
            "last_seen": point_time
        })

    print(f"Online: {online_count}, Offline: {offline_count}, Total processed: {online_count + offline_count}")
    return {
        "online": online_count,
        "offline": 542 - online_count,
        "total": online_count + offline_count,
        "details": details
    }


@app.route("/api/system-status")
def system_status():
    return jsonify(get_system_online_status())

@app.route("/api/low-usage")
def get_low_usage_systems():
    """
    Returns systems with CPU usage < 4%
    """
    query = """
    SELECT LAST("usage_idle") AS usage_idle
    FROM "cpu"
    WHERE customer_name = 'MCE'
    GROUP BY host, floor_name
    """

    result = client.query(query)
    now = datetime.now(timezone.utc)
    low_usage_systems = []

    for (_, tags), points in result.items():
        point = next(points, None)
        if point is None:
            continue

        host = tags.get("host", "UNKNOWN")
        floor = tags.get("floor_name", "UNKNOWN")
        
        usage_idle = point.get("usage_idle")
        point_time = point.get("time")
        
        if usage_idle is None or point_time is None:
            continue

        # Check if data is fresh (within 5 minutes)
        try:
            if isinstance(point_time, str):
                if point_time.endswith('Z'):
                    point_time_dt = datetime.fromisoformat(point_time.replace("Z", "+00:00"))
                else:
                    point_time_dt = datetime.fromisoformat(point_time)
                    if point_time_dt.tzinfo is None:
                        point_time_dt = point_time_dt.replace(tzinfo=timezone.utc)
            else:
                point_time_dt = point_time
                if point_time_dt.tzinfo is None:
                    point_time_dt = point_time_dt.replace(tzinfo=timezone.utc)
        except:
            continue

        time_diff = now - point_time_dt
        if time_diff > ONLINE_THRESHOLD:
            continue

        cpu_usage = 100 - usage_idle
        
        # Flag systems with CPU usage < 4%
        if cpu_usage < 4:
            low_usage_systems.append({
                "host": host,
                "floor_name": floor,
                "cpu_usage": round(cpu_usage, 2),
                "last_seen": point_time
            })

    return jsonify(low_usage_systems)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)