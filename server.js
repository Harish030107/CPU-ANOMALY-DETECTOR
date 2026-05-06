import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import si from 'systeminformation';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { RandomForestPredictor } from './predictor.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer);

const WINDOW_SIZE = 10;
const FEATURES = ['cpu_usage', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max', 'delta'];

let model = null;
let cpuHistory = [];

// Load trained model
function loadModel() {
    const modelPath = path.join(__dirname, 'rf_freq_model.json');
    if (!fs.existsSync(modelPath)) {
        console.error('[ERROR] Model file not found. Run "npm run convert" first to convert the .pkl model.');
        process.exit(1);
    }
    model = RandomForestPredictor.load(modelPath);
    console.log('[OK] RandomForest model loaded successfully');
    console.log(`[INFO] Trees: ${model.nEstimators}, Features: ${model.nFeatures}`);
}

// Compute rolling statistics
function computeRollingStats(values, windowSize) {
    const n = values.length;
    if (n < windowSize) return null;

    const window = values.slice(-windowSize);
    const mean = window.reduce((a, b) => a + b, 0) / window.length;
    const variance = window.reduce((a, b) => a + (b - mean) ** 2, 0) / window.length;
    const std = Math.sqrt(variance);
    const min = Math.min(...window);
    const max = Math.max(...window);

    return { mean, std, min, max };
}

// Serve dashboard
app.use(express.static(__dirname));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'dashboard.html'));
});

// Socket.IO connection
io.on('connection', (socket) => {
    console.log('[INFO] Client connected');

    socket.on('disconnect', () => {
        console.log('[INFO] Client disconnected');
    });
});

// Collect and predict CPU usage
async function monitorCPU() {
    try {
        const load = await si.currentLoad();
        const cpu = load.currentLoad;

        cpuHistory.push(cpu);

        // Keep history limited
        if (cpuHistory.length > 100) {
            cpuHistory.shift();
        }

        let prediction = 0;

        // Only predict if we have enough history
        if (cpuHistory.length >= WINDOW_SIZE) {
            const stats = computeRollingStats(cpuHistory, WINDOW_SIZE);
            const delta = cpuHistory.length > 1 ? cpu - cpuHistory[cpuHistory.length - 2] : 0;

            const features = [
                cpu,
                stats.mean,
                stats.std,
                stats.min,
                stats.max,
                delta
            ];

            // Predict using the model
            prediction = model.predict(features);
        }

        const data = {
            timestamp: new Date().toISOString(),
            cpu_usage: cpu,
            prediction: prediction
        };

        // Emit to all connected clients
        io.emit('cpu-data', data);

        console.log(`[${new Date().toLocaleTimeString()}] CPU: ${cpu.toFixed(2)}% | Prediction: ${prediction === 1 ? 'ANOMALY' : 'Normal'}`);
    } catch (error) {
        console.error('[ERROR]', error);
        io.emit('error', { message: error.message });
    }
}

// Start server
const PORT = process.env.PORT || 3000;

loadModel();

httpServer.listen(PORT, () => {
    console.log(`[OK] Dashboard server running at http://localhost:${PORT}`);
    console.log('[INFO] Open your browser to view the live dashboard');
    console.log('[INFO] Press Ctrl+C to stop');

    // Start monitoring
    setInterval(monitorCPU, 1000);
});
