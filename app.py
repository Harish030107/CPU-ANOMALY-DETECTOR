from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import psutil
import joblib
import numpy as np
import os
import time
from datetime import datetime
from collections import deque
import threading

app = Flask(__name__, template_folder='.')
app.config['SECRET_KEY'] = 'cpu-anomaly-detection-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Configuration
WINDOW_SIZE = 10
FEATURES = ['cpu_usage', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max', 'delta']
MODEL_PATH = 'rf_labeled_model.pkl'

# Global variables
model = None
cpu_history = deque(maxlen=100)
monitoring_active = False

def load_model():
    """Load the trained RandomForest model"""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("[INFO] Please ensure rf_labeled_model.pkl exists in the current directory")
        return False
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded successfully from {MODEL_PATH}")
        print(f"[INFO] Model type: {type(model).__name__}")
        print(f"[INFO] Number of estimators: {model.n_estimators}")
        print(f"[INFO] Number of features: {model.n_features_in_}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False

def compute_rolling_stats(values, window_size):
    """Compute rolling statistics for the given values"""
    if len(values) < window_size:
        return None
    
    window = list(values)[-window_size:]
    mean = np.mean(window)
    std = np.std(window)
    minimum = np.min(window)
    maximum = np.max(window)
    
    return {
        'mean': mean,
        'std': std,
        'min': minimum,
        'max': maximum
    }

def monitor_cpu():
    """Monitor CPU usage and make predictions"""
    global cpu_history, monitoring_active
    
    print("[INFO] Starting CPU monitoring...")
    
    while monitoring_active:
        try:
            # Get CPU usage
            cpu = psutil.cpu_percent(interval=1)
            cpu_history.append(cpu)
            
            prediction = 0  # Default: normal
            
            # Only predict if we have enough history
            if len(cpu_history) >= WINDOW_SIZE:
                stats = compute_rolling_stats(cpu_history, WINDOW_SIZE)
                delta = cpu - cpu_history[-2] if len(cpu_history) > 1 else 0
                
                # Prepare features with proper column names
                features = np.array([[
                    cpu,
                    stats['mean'],
                    stats['std'],
                    stats['min'],
                    stats['max'],
                    delta
                ]])
                
                # Make prediction (suppress sklearn warnings)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prediction = int(model.predict(features)[0])
            
            # Prepare data packet
            data = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': float(cpu),
                'prediction': prediction
            }
            
            # Emit to all connected clients
            socketio.emit('cpu-data', data, namespace='/')
            
            # Log prediction
            status = 'ANOMALY' if prediction == 1 else 'Normal'
            print(f"[{datetime.now().strftime('%I:%M:%S %p')}] CPU: {cpu:.2f}% | Prediction: {status}")
            
        except Exception as e:
            print(f"[ERROR] Monitoring error: {e}")
            socketio.emit('error', {'message': str(e)}, namespace='/')
            time.sleep(1)

@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template('dashboard_flask.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('[INFO] Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('[INFO] Client disconnected')

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start CPU monitoring"""
    global monitoring_active
    if not monitoring_active:
        monitoring_active = True
        thread = threading.Thread(target=monitor_cpu, daemon=True)
        thread.start()
        print('[INFO] Monitoring started')

if __name__ == '__main__':
    print("=" * 60)
    print("CPU Anomaly Detection Dashboard - Flask Edition")
    print("=" * 60)
    
    # Load model
    if not load_model():
        print("[ERROR] Cannot start server without model")
        exit(1)
    
    # Start monitoring in background
    monitoring_active = True
    monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
    monitor_thread.start()
    
    # Start server
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[OK] Dashboard server starting on http://localhost:{port}")
    print("[INFO] Open your browser to view the live dashboard")
    print("[INFO] Press Ctrl+C to stop\n")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
