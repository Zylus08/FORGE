"""M3 Controlled Ablation Study Runner.
Runs paired, isolated benchmarks across architecture configurations and logs to CSV.
"""
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forge.benchmarking.canonical_runner import run_benchmark_subprocess

def main():
    out_dir = PROJECT_ROOT / "benchmarks" / "m3_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # 1. Official Certified Stock Competitor (Hurdle)
        {"id": "A0_Stock_Eager_NHWC_T10", "backend": "torch-eager", "layout": "NHWC", "threads": 10},
        # 2. Stock Eager with NCHW layout (shows layout impact)
        {"id": "A1_Stock_Eager_NCHW_T10", "backend": "torch-eager", "layout": "NCHW", "threads": 10},
        # 3. Stock Tuned (NHWC, T10)
        {"id": "A2_Stock_Tuned_NHWC_T10", "backend": "torch-tuned", "layout": "NHWC", "threads": 10},
        # 4. FORGE M3 Zero-Mask (NHWC, T10)
        {"id": "A3_Forge_M3_ZeroMask_NHWC_T10", "backend": "forge", "layout": "NHWC", "threads": 10},
        # 5. FORGE M3 Zero-Mask (NCHW, T10)
        {"id": "A4_Forge_M3_ZeroMask_NCHW_T10", "backend": "forge", "layout": "NCHW", "threads": 10},
    ]

    results = []
    print("=" * 80)
    print("STARTING M3 ABLATION RUNS (30 timed steps per config, isolated subprocesses)")
    print("=" * 80)

    for exp in experiments:
        exp_id = exp["id"]
        json_path = out_dir / f"{exp_id}.json"
        print(f"\n---> Running {exp_id}...")
        try:
            data = run_benchmark_subprocess(
                backend=exp["backend"],
                layout=exp["layout"],
                threads=exp["threads"],
                epochs=1,
                warmup_steps=5,
                timed_steps=30,
                output_json=str(json_path)
            )
            ep_data = data["epochs_data"][0]
            print(f"     [OK] Median Step: {ep_data['median_step_ms']} ms | Fwd: {ep_data['median_fwd_ms']} ms | Bwd: {ep_data['median_bwd_ms']} ms | Samples/s: {ep_data['samples_per_sec']}")
            results.append({
                "experiment_id": exp_id,
                "backend": exp["backend"],
                "layout": exp["layout"],
                "threads": exp["threads"],
                "median_step_ms": ep_data["median_step_ms"],
                "mean_step_ms": ep_data["mean_step_ms"],
                "p95_step_ms": ep_data["p95_step_ms"],
                "median_fwd_ms": ep_data["median_fwd_ms"],
                "median_bwd_ms": ep_data["median_bwd_ms"],
                "median_opt_ms": ep_data["median_opt_ms"],
                "median_data_ms": ep_data["median_data_ms"],
                "samples_per_sec": ep_data["samples_per_sec"],
                "peak_rss_mb": data["peak_rss_mb"],
                "json_path": str(json_path),
            })
        except Exception as e:
            print(f"     [FAILED] {e}")

    # Write ABLATION_RESULTS.csv
    csv_path = PROJECT_ROOT / "audit" / "m3" / "ABLATION_RESULTS.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "experiment_id", "backend", "layout", "threads",
            "median_step_ms", "mean_step_ms", "p95_step_ms",
            "median_fwd_ms", "median_bwd_ms", "median_opt_ms",
            "median_data_ms", "samples_per_sec", "peak_rss_mb", "json_path"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nWrote ablation results to {csv_path}")

    # Write BENCHMARK_RESULTS.csv
    bench_csv = PROJECT_ROOT / "audit" / "m3" / "BENCHMARK_RESULTS.csv"
    with open(bench_csv, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "configuration", "backend", "layout", "threads",
            "median_step_ms", "speedup_vs_certified_stock", "samples_per_sec", "status"
        ])
        writer.writeheader()
        hurdle_time = None
        for r in results:
            if r["experiment_id"] == "A0_Stock_Eager_NHWC_T10":
                hurdle_time = r["median_step_ms"]
                break

        for r in results:
            speedup = round(hurdle_time / r["median_step_ms"], 4) if hurdle_time else 1.0
            status = "CERTIFIED_BASELINE" if "A0" in r["experiment_id"] else ("SLOWER" if speedup < 1.0 else "FASTER")
            writer.writerow({
                "configuration": r["experiment_id"],
                "backend": r["backend"],
                "layout": r["layout"],
                "threads": r["threads"],
                "median_step_ms": r["median_step_ms"],
                "speedup_vs_certified_stock": f"{speedup:.4f}x",
                "samples_per_sec": r["samples_per_sec"],
                "status": status,
            })
    print(f"Wrote benchmark results to {bench_csv}")

if __name__ == "__main__":
    main()
