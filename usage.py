from influxdb import InfluxDBClient
import pandas as pd

# -----------------------------
# InfluxDB Connection
# -----------------------------
client = InfluxDBClient(
    host="195.201.164.158",  
    port=9086,  
    database="end_user_monitoring"
)

# -----------------------------
# InfluxDB Query
# -----------------------------
query = """
      SELECT LAST("usage_user") AS cpu, host, time  
FROM cpu  
WHERE customer_name =~ /MCE/  
GROUP BY host
    """

print(client.ping())

'''result = client.query(query)
    

# -----------------------------
# Convert to DataFrame
# -----------------------------
points = list(result.get_points())
df = pd.DataFrame(points)

# -----------------------------
# Display Table
# -----------------------------
print(df.head())'''