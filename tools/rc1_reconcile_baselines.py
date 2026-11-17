"""RC1 Baseline Reconciliation Runner: Counterbalanced Repeated Runs.
Evaluates Stock Eager NHWC vs NCHW and FORGE NHWC vs NCHW.
Saves raw per-step measurements and statistical significance tests.
"""
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from forge.benchmarking.canonical_runner import run_benchmark_subprocess

def calc_stats(data_list):
    arr = np.array(data_list, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    # Student's t 95% critical value for small n
    t_crit = {
        1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042
    }.get(n - 1, 1.96)
    ci95_low = mean - t_crit * sem
    ci95_high = mean + t_crit * sem
    return {
        "n": n,
        "mean": round(mean, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "sem": round(sem, 2),
        "ci95_low": round(ci95_low, 2),
        "ci95_high": round(ci95_high, 2),
    }

def welch_t_test(x, y):
    """Welch's t-test for unequal variances."""
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n1, n2 = len(x), len(y)
    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    
    denom = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / denom if denom > 0 else 0.0
    
    # Welch-Satterthwaite degrees of freedom
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    
    return float(t_stat), float(df)

def main():
    out_dir = PROJECT_ROOT / "benchmarks" / "rc1_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Counterbalanced schedule for Stock Eager
    # 10 alternating runs: NHWC, NCHW, NCHW, NHWC, NHWC, NCHW, NCHW, NHWC, NHWC, NCHW
    stock_schedule = [
        ("stock_nhwc_r1", "torch-eager", "NHWC"),
        ("stock_nchw_r1", "torch-eager", "NCHW"),
        ("stock_nchw_r2", "torch-eager", "NCHW"),
        ("stock_nhwc_r2", "torch-eager", "NHWC"),
        ("stock_nhwc_r3", "torch-eager", "NHWC"),
        ("stock_nchw_r3", "torch-eager", "NCHW"),
        ("stock_nchw_r4", "torch-eager", "NCHW"),
        ("stock_nhwc_r4", "torch-eager", "NHWC"),
        ("stock_nhwc_r5", "torch-eager", "NHWC"),
        ("stock_nchw_r5", "torch-eager", "NCHW"),
    ]

    forge_schedule = [
        ("forge_nhwc_r1", "forge", "NHWC"),
        ("forge_nchw_r1", "forge", "NCHW"),
        ("forge_nchw_r2", "forge", "NCHW"),
        ("forge_nhwc_r2", "forge", "NHWC"),
    ]

    raw_results = {}
    print("=" * 80)
    print("PROJECT FORGE — RC1 BASELINE RECONCILIATION & CONFIRMATION BENCHMARK")
    print("=" * 80)

    # 1. Run Stock Eager Counterbalanced Suite
    for run_id, backend, layout in stock_schedule:
        print(f"---> Executing {run_id} ({backend}, {layout}, T10)...")
        json_file = out_dir / f"{run_id}.json"
        res = run_benchmark_subprocess(
            backend=backend,
            layout=layout,
            threads=10,
            epochs=1,
            warmup_steps=5,
            timed_steps=20,
            output_json=str(json_file),
            skip_eval=True
        )
        ep = res["epochs_data"][0]
        print(f"     [OK] Median: {ep['median_step_ms']} ms | Fwd: {ep['median_fwd_ms']} ms | Bwd: {ep['median_bwd_ms']} ms | Samples/s: {ep['samples_per_sec']}")
        raw_results[run_id] = {
            "backend": backend,
            "layout": layout,
            "median_step_ms": ep["median_step_ms"],
            "mean_step_ms": ep["mean_step_ms"],
            "median_fwd_ms": ep["median_fwd_ms"],
            "median_bwd_ms": ep["median_bwd_ms"],
            "samples_per_sec": ep["samples_per_sec"],
        }
        time.sleep(2)  # Cooldown between runs to normalize thermal state

    # 2. Run FORGE Counterbalanced Suite
    for run_id, backend, layout in forge_schedule:
        print(f"---> Executing {run_id} ({backend}, {layout}, T10)...")
        json_file = out_dir / f"{run_id}.json"
        res = run_benchmark_subprocess(
            backend=backend,
            layout=layout,
            threads=10,
            epochs=1,
            warmup_steps=5,
            timed_steps=20,
            output_json=str(json_file),
            skip_eval=True
        )
        ep = res["epochs_data"][0]
        print(f"     [OK] Median: {ep['median_step_ms']} ms | Fwd: {ep['median_fwd_ms']} ms | Bwd: {ep['median_bwd_ms']} ms | Samples/s: {ep['samples_per_sec']}")
        raw_results[run_id] = {
            "backend": backend,
            "layout": layout,
            "median_step_ms": ep["median_step_ms"],
            "mean_step_ms": ep["mean_step_ms"],
            "median_fwd_ms": ep["median_fwd_ms"],
            "median_bwd_ms": ep["median_bwd_ms"],
            "samples_per_sec": ep["samples_per_sec"],
        }
        time.sleep(2)

    # 3. Aggregate and Analyze
    stock_nhwc_meds = [raw_results[f"stock_nhwc_r{i}"]["median_step_ms"] for i in range(1, 6)]
    stock_nchw_meds = [raw_results[f"stock_nchw_r{i}"]["median_step_ms"] for i in range(1, 6)]
    forge_nhwc_meds = [raw_results[f"forge_nhwc_r{i}"]["median_step_ms"] for i in range(1, 3)]
    forge_nchw_meds = [raw_results[f"forge_nchw_r{i}"]["median_step_ms"] for i in range(1, 3)]

    stats_stock_nhwc = calc_stats(stock_nhwc_meds)
    stats_stock_nchw = calc_stats(stock_nchw_meds)
    stats_forge_nhwc = calc_stats(forge_nhwc_meds)
    stats_forge_nchw = calc_stats(forge_nchw_meds)

    t_val, df = welch_t_test(stock_nhwc_meds, stock_nchw_meds)

    summary = {
        "stock_nhwc": {
            "raw_medians_ms": stock_nhwc_meds,
            "stats": stats_stock_nhwc,
        },
        "stock_nchw": {
            "raw_medians_ms": stock_nchw_meds,
            "stats": stats_stock_nchw,
        },
        "forge_nhwc": {
            "raw_medians_ms": forge_nhwc_meds,
            "stats": stats_forge_nhwc,
        },
        "forge_nchw": {
            "raw_medians_ms": forge_nchw_meds,
            "stats": stats_forge_nchw,
        },
        "statistical_comparison_stock": {
            "welch_t_statistic": round(t_val, 4),
            "degrees_of_freedom": round(df, 2),
            "diff_of_means_ms": round(stats_stock_nhwc["mean"] - stats_stock_nchw["mean"], 2),
            "pct_diff": round((stats_stock_nhwc["mean"] - stats_stock_nchw["mean"]) / stats_stock_nhwc["mean"] * 100, 2),
            "statistically_distinguishable_alpha_0_05": abs(t_val) > 2.306,
        }
    }

    summary_file = out_dir / "rc1_baseline_reconciliation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("RECONCILIATION SUMMARY RESULTS:")
    print("=" * 80)
    print(f"Stock NHWC (5 runs): Mean of Medians = {stats_stock_nhwc['mean']} ms ± {stats_stock_nhwc['sem']} ms [95% CI: {stats_stock_nhwc['ci95_low']} - {stats_stock_nhwc['ci95_high']}]")
    print(f"Stock NCHW (5 runs): Mean of Medians = {stats_stock_nchw['mean']} ms ± {stats_stock_nchw['sem']} ms [95% CI: {stats_stock_nchw['ci95_low']} - {stats_stock_nchw['ci95_high']}]")
    print(f"FORGE NHWC (2 runs): Mean of Medians = {stats_forge_nhwc['mean']} ms | Median: {stats_forge_nhwc['median']} ms")
    print(f"FORGE NCHW (2 runs): Mean of Medians = {stats_forge_nchw['mean']} ms | Median: {stats_forge_nchw['median']} ms")
    print(f"Welch's t-stat: {t_val:.4f} (df={df:.2f})")
    print(f"Statistically distinguishable at alpha=0.05: {summary['statistical_comparison_stock']['statistically_distinguishable_alpha_0_05']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
