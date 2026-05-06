import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { DecisionTreeClassifier } from 'ml-cart';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FEATURES = [
    'cpu_usage',
    'rolling_mean',
    'rolling_std',
    'rolling_min',
    'rolling_max',
    'delta',
];

function loadCsv(filePath) {
    const text = fs.readFileSync(filePath, 'utf8');
    const lines = text.trim().split(/\r?\n/);
    const header = lines[0].split(',');
    const rows = lines.slice(1).map((line) => {
        const cols = line.split(',');
        const obj = {};
        header.forEach((h, i) => {
            obj[h] = cols[i];
        });
        return obj;
    });
    return { header, rows };
}

function toDataset(rows) {
    const X = [];
    const y = [];
    for (const r of rows) {
        // skip rows with missing values
        if (!FEATURES.every((f) => r[f] !== undefined && r[f] !== '')) continue;
        const vec = FEATURES.map((f) => Number(r[f]));
        if (vec.some((v) => Number.isNaN(v))) continue;
        const label = Number(r.label);
        if (Number.isNaN(label)) continue;
        X.push(vec);
        y.push(label);
    }
    return { X, y };
}

function saveModel(model, outPath) {
    const json = model.toJSON();
    fs.writeFileSync(outPath, JSON.stringify(json, null, 2));
}

async function main() {
    // Prefer the file produced by 4.js, else fall back to the provided CSV
    const candidates = [
        path.join(__dirname, 'cpu_training_data_freq.csv'),
        path.join(__dirname, 'cpu_training_data_freq_minmax.csv'),
    ];
    const csvPath = candidates.find((p) => fs.existsSync(p));
    if (!csvPath) {
        console.error('[ERROR] No CSV dataset found. Run "npm run start:4" first.');
        process.exit(1);
    }
    console.log(`[INFO] Loading dataset: ${path.basename(csvPath)}`);
    const { rows } = loadCsv(csvPath);
    const { X, y } = toDataset(rows);
    if (X.length === 0) {
        console.error('[ERROR] Dataset is empty after preprocessing.');
        process.exit(1);
    }
    console.log(`[INFO] Samples: ${X.length}, Features: ${FEATURES.length}`);

    const options = {
        gainFunction: 'gini',
        maxDepth: 10,
        minNumSamples: 3,
    };

    const dt = new DecisionTreeClassifier(options);
    dt.train(X, y);

    const modelPath = path.join(__dirname, 'dt_freq_model.json');
    const json = dt.toJSON();
    fs.writeFileSync(modelPath, JSON.stringify(json, null, 2));
    console.log(`[OK] Model trained and saved as ${path.basename(modelPath)}`);
}

main().catch((err) => {
    console.error('[ERROR]', err);
    process.exit(1);
});
