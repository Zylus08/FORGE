"""Tests for benchmark harness output format, columns, and exclusions."""
import csv
import math
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
AUDIT_DIR = PROJECT_ROOT / "audit" / "m1"


def test_benchmark_csv_output_columns():
    """Verify that all generated benchmark CSV files contain the required schema columns."""
    required_cols = {
        "backend",
        "epoch",
        "epoch_time_s",
        "samples_per_sec",
        "train_accuracy_pct",
        "test_accuracy_pct",
        "test_loss",
        "median_step_time_ms",
        "p95_step_time_ms",
        "median_data_time_ms",
        "median_fwd_time_ms",
        "median_bwd_time_ms",
        "median_opt_time_ms",
        "num_steps_timed",
    }

    csv_files = [
        RESULTS_DIR / "baseline_eager.csv",
        RESULTS_DIR / "baseline_tuned.csv",
        RESULTS_DIR / "baseline_compiled.csv",
        RESULTS_DIR / "forge_results.csv",
        AUDIT_DIR / "BASELINE_RESULTS.csv",
    ]

    for csv_file in csv_files:
        assert csv_file.exists(), f"Benchmark result CSV not found: {csv_file}"
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = required_cols - fieldnames
            assert not missing, f"File {csv_file.name} missing required columns: {missing}"


def test_timing_measurements_are_positive_and_finite():
    """Verify that timing metrics and throughput values are strictly positive and finite."""
    csv_files = [
        RESULTS_DIR / "baseline_eager.csv",
        RESULTS_DIR / "baseline_tuned.csv",
        RESULTS_DIR / "baseline_compiled.csv",
        RESULTS_DIR / "forge_results.csv",
    ]

    for csv_file in csv_files:
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0, f"No data in {csv_file.name}"
            for row in rows:
                samples_per_sec = float(row["samples_per_sec"])
                epoch_time = float(row["epoch_time_s"])
                median_step = float(row["median_step_time_ms"])
                p95_step = float(row["p95_step_time_ms"])

                assert math.isfinite(samples_per_sec) and samples_per_sec > 0
                assert math.isfinite(epoch_time) and epoch_time > 0
                assert math.isfinite(median_step) and median_step > 0
                assert math.isfinite(p95_step) and p95_step >= median_step


def test_warmup_steps_excluded_from_reported_timing():
    """Verify that reported num_steps_timed excludes warm-up steps.

    CIFAR-10 has 50,000 training images. With batch_size=128, total steps per epoch = ceil(50000/128) = 391.
    For epoch 1, warm-up steps (2 steps) are discarded, so num_steps_timed should be 388 or 389.
    For subsequent epochs, 390 or 391 steps are timed.
    """
    csv_files = [
        RESULTS_DIR / "baseline_eager.csv",
        RESULTS_DIR / "baseline_tuned.csv",
        RESULTS_DIR / "baseline_compiled.csv",
        RESULTS_DIR / "forge_results.csv",
    ]

    for csv_file in csv_files:
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Find epoch 1
            epoch1_rows = [r for r in rows if r["epoch"] == "1"]
            if epoch1_rows:
                timed_e1 = int(epoch1_rows[0]["num_steps_timed"])
                # Total steps is 391, warmup is 2, so timed steps must be < 391 (specifically 388 or 389)
                assert timed_e1 < 391, f"Epoch 1 num_steps_timed ({timed_e1}) indicates warmup steps were not excluded"
