from flask import Flask, render_template, request,jsonify 
from influxdb import InfluxDBClient
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# -------------------------------
# InfluxDB Configuration
# -------------------------------
client = InfluxDBClient(
    host="195.201.164.158",  
    port=9086,  
    database="end_user_monitoring"
)

@app.route('/')
def index():
    query = """
    SELECT COUNT("cpu_usage") AS anomaly_count          
    FROM cpu
    """
    result = client.query(query)
    with open('index2.txt') as f:
        f.write(str(result))
    print(result)