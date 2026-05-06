# Node.js Scripts (4.js and 5.js)

This project provides Node.js equivalents for CPU data collection and simple anomaly labeling, plus a basic classifier training step.

## Scripts
- 4.js: Collects CPU usage for 120 seconds, computes rolling features, derives a frequent-value normal region, labels anomalies, and saves cpu_training_data_freq.csv.
- 5.js: Loads the CSV dataset and trains a Decision Tree classifier (ml-cart), saving dt_freq_model.json.

## Setup
```bash
npm install
```

## Run
- Collect data (takes ~2 minutes):
```bash
npm run start:4
```
- Train model (uses the CSV produced by 4.js or falls back to cpu_training_data_freq_minmax.csv if present):
```bash
npm run start:5
```

## Notes
- 4.js uses the systeminformation package to get CPU load; ensure Node.js is installed and accessible.
- 5.js uses ml-cart for a lightweight Decision Tree; the saved JSON can be reloaded to make predictions in Node later.