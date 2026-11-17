"""Executes isolated canonical baseline sweeps for Milestone M3.
Every run is isolated in its own subprocess.
"""
import sys
import os
from pathlib import Path
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forge.benchmarking.canonical_runner import run_benchmark_subprocess

OUTPUT_DIR = PROJECT_ROOT / "benchmarks" / "m3_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    # Primary tuned candidates
    {"backend": "torch-tuned", "layout": "NHWC", "threads": 10, "interop": 1, "timed_steps": 30},
    {"backend": "torch-tuned", "layout": "NHWC", "threads": 8, "interop": 1, "timed_steps": 30},
    {"backend": "torch-tuned", "layout": "NHWC", "threads": 6, "interop": 1, "timed_steps": 30},
    {"backend": "torch-tuned", "layout": "NHWC", "threads": 12, "interop": 1, "timed_steps": 30},
    {"backend": "torch-tuned", "layout": "NHWC", "threads": 16, "interop": 1, "timed_steps": 30},
    # Un-tuned and NCHW references
    {"backend": "torch-eager", "layout": "NCHW", "threads": 10, "interop": 1, "timed_steps": 30},
    {"backend": "torch-eager", "layout": "NHWC", "threads": 10, "interop": 1, "timed_steps": 30},
    # Compiled baseline
    {"backend": "torch-compiled", "layout": "NHWC", "threads": 10, "interop": 1, "timed_steps": 20},
]

manifest = []

for c in configs:
    tag = f"{c['backend']}_{c['layout']}_T{c['threads']}"
    out_json = OUTPUT_DIR / f"{tag}.json"
    print(f"Running baseline: {tag} ...", flush=True)
    try:
        data = run_benchmark_subprocess(
            backend=c["backend"],
            layout=c["layout"],
            threads=c["threads"],
            interop=c["interop"],
            epochs=1,
            warmup_steps=5,
            timed_steps=c.get("timed_steps", 30),
            output_json=str(out_json),
        )
        ep = data["epochs_data"][0]
        summary = {
            "backend": data["backend"],
            "layout": data["layout"],
            "threads": data["threads"],
            "interop": data["interop"],
            "compile_overhead_s": data["compile_overhead_s"],
            "median_step_ms": ep["median_step_ms"],
            "mean_step_ms": ep["mean_step_ms"],
            "p95_step_ms": ep["p95_step_ms"],
            "median_fwd_ms": ep["median_fwd_ms"],
            "median_bwd_ms": ep["median_bwd_ms"],
            "median_opt_ms": ep["median_opt_ms"],
            "median_data_ms": ep["median_data_ms"],
            "samples_per_sec": ep["samples_per_sec"],
            "peak_rss_mb": data["peak_rss_mb"],
            "json_path": str(out_json),
        }
        manifest.append(summary)
        print(f"  -> median={ep['median_step_ms']:.1f}ms, fwd={ep['median_fwd_ms']:.1f}ms, bwd={ep['median_bwd_ms']:.1f}ms, smp/s={ep['samples_per_sec']:.1f}", flush=True)
    except Exception as e:
        print(f"  [ERROR] {tag} failed: {e}", flush=True)

# Save baseline summary CSV
out_csv = PROJECT_ROOT / "audit" / "m3" / "baseline_tuning_results.csv"
if manifest:
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\nBaseline tuning summary saved to: {out_csv}")
