import fs from 'fs';
import si from 'systeminformation';

const COLLECT_SECONDS = 120;
const WINDOW_SIZE = 10;
const TOP_PERCENT = 0.9; // Keep values that cover 90% of frequency

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function rollingStats(values, window) {
    const n = values.length;
    const mean = Array(n).fill(null);
    const std = Array(n).fill(null);
    const min = Array(n).fill(null);
    const max = Array(n).fill(null);
    for (let i = 0; i < n; i++) {
        const start = i - window + 1;
        if (start < 0) continue;
        const slice = values.slice(start, i + 1);
        const m = slice.reduce((a, b) => a + b, 0) / slice.length;
        const v = slice.reduce((a, b) => a + (b - m) * (b - m), 0) / slice.length;
        mean[i] = m;
        std[i] = Math.sqrt(v);
        min[i] = Math.min(...slice);
        max[i] = Math.max(...slice);
    }
    return { mean, std, min, max };
}

function histogram(values, binEdges) {
    const counts = Array(binEdges.length - 1).fill(0);
    for (const v of values) {
        // find bin index
        if (v < binEdges[0] || v > binEdges[binEdges.length - 1]) continue;
        // last edge inclusive logic mirrors numpy where last bin right edge excluded except final
        let idx = Math.floor((v - binEdges[0]) / (binEdges[1] - binEdges[0]));
        if (idx >= counts.length) idx = counts.length - 1;
        if (idx < 0) idx = 0;
        counts[idx] += 1;
    }
    return counts;
}

async function collectCpuUsage(seconds) {
    const data = [];
    for (let i = 0; i < seconds; i++) {
        const load = await si.currentLoad();
        const cpu = load.currentLoad; // percentage
        data.push({ Timestamp: new Date().toISOString(), cpu_usage: cpu });
        await sleep(1000);
    }
    return data;
}

async function main() {
    console.log(`[INFO] Collecting CPU usage for ${COLLECT_SECONDS}s ...`);
    const rows = await collectCpuUsage(COLLECT_SECONDS);
    const cpu = rows.map((r) => r.cpu_usage);

    const { mean, std, min, max } = rollingStats(cpu, WINDOW_SIZE);
    const delta = [null];
    for (let i = 1; i < cpu.length; i++) delta[i] = cpu[i] - cpu[i - 1];

    // Drop rows with null (before window filled)
    const startIdx = WINDOW_SIZE - 1;
    const filtered = [];
    for (let i = 0; i < rows.length; i++) {
        if (i < startIdx) continue;
        filtered.push({
            Timestamp: rows[i].Timestamp,
            cpu_usage: cpu[i],
            rolling_mean: mean[i],
            rolling_std: std[i],
            rolling_min: min[i],
            rolling_max: max[i],
            delta: delta[i],
        });
    }

    // Build frequency distribution (0.5% bins)
    const maxCpu = Math.max(...filtered.map((r) => r.cpu_usage));
    const step = 0.5;
    const binEdges = [];
    for (let v = 0; v <= Math.ceil(maxCpu + 1); v += step) binEdges.push(v);
    const hist = histogram(filtered.map((r) => r.cpu_usage), binEdges);

    // Sort bins by frequency and cover TOP_PERCENT of total
    const total = hist.reduce((a, b) => a + b, 0);
    const indices = hist.map((c, i) => [c, i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
    let cumulative = 0;
    const selected = [];
    for (const idx of indices) {
        selected.push(idx);
        cumulative += hist[idx];
        if (total > 0 && cumulative / total >= TOP_PERCENT) break;
    }
    const normalMin = binEdges[Math.min(...selected)];
    const normalMax = binEdges[Math.max(...selected) + 1];
    console.log(`[INFO] Normal region based on frequent values: ${normalMin.toFixed(2)} - ${normalMax.toFixed(2)}`);

    // Label anomalies
    for (const r of filtered) {
        r.label = r.cpu_usage < normalMin || r.cpu_usage > normalMax ? 1 : 0;
    }

    // Save CSV
    const headers = [
        'Timestamp',
        'cpu_usage',
        'rolling_mean',
        'rolling_std',
        'rolling_min',
        'rolling_max',
        'delta',
        'label',
    ];
    const lines = [headers.join(',')];
    for (const r of filtered) {
        lines.push([
            r.Timestamp,
            r.cpu_usage,
            r.rolling_mean,
            r.rolling_std,
            r.rolling_min,
            r.rolling_max,
            r.delta,
            r.label,
        ].join(','));
    }
    fs.writeFileSync('cpu_training_data_freq.csv', lines.join('\n'));
    console.log('[OK] Data collected and saved as cpu_training_data_freq.csv');
}

main().catch((err) => {
    console.error('[ERROR]', err);
    process.exit(1);
});
