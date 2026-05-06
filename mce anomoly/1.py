import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------------------------------
# Email Configuration
# -------------------------------------------------
SENDER_EMAIL = "signinsih@gmail.com"
RECEIVER_EMAIL = "s.lokeeshkumar006@gmail.com"
EMAIL_PASSWORD = "ydrjjgslvimnbqqu"

def send_anomaly_email():
    subject = "CPU Anomaly Alert – Sustained for 5 Minutes"

    body = f"""
CPU Anomaly Detected (Sustained)

Host       : desktop-1234
Floor      : 1F
CPU Usage  : 80.00
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
