"""Run complete M2 Benchmark Suite across all 4 configurations for 5 epochs each."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import subprocess
import csv
from pathlib import Path

PYTHON = r"C:\Users\mishr\anaconda3\envs\reswork\python.exe"
OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    ("torch-eager", "m2_baseline_eager.csv"),
    ("torch-tuned", "m2_baseline_tuned.csv"),
    ("torch-compiled", "m2_baseline_compiled.csv"),
    ("forge", "m2_forge_results.csv"),
]

all_rows = []

for backend, filename in configs:
    out_csv = OUTPUT_DIR / filename
    print(f"\n{'='*70}")
    print(f"RUNNING M2 BENCHMARK: {backend.upper()} (5 Epochs)")
    print(f"{'='*70}\n")
    
    cmd = [
        PYTHON, "-m", "forge.benchmarking.run_baselines",
        "--backend", backend,
        "--epochs", "5",
        "--output-csv", str(out_csv),
    ]
    
    res = subprocess.run(cmd, capture_output=False, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Benchmark failed for {backend} with return code {res.returncode}")
    else:
        print(f"[SUCCESS] Benchmark completed for {backend}")
        if out_csv.exists():
            with open(out_csv, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                all_rows.extend(rows)

# Write aggregated BENCHMARK_RESULTS.csv
agg_csv = Path("audit/m2/BENCHMARK_RESULTS.csv")
agg_csv.parent.mkdir(parents=True, exist_ok=True)
if all_rows:
    fieldnames = list(all_rows[0].keys())
    with open(agg_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAggregated results written to {agg_csv}")
