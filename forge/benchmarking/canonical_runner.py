"""FORGE Canonical Benchmark Runner for Milestone M3.
Executes benchmarks in isolated subprocesses with matched measurement boundaries.
Strictly preserves workload contract: ResNet-20, CIFAR-10, batch size 128, FP32, SGD.
"""
import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe environment configuration on Drive E:
os.environ["TMP"] = r"E:\FORGE\.tmp"
os.environ["TEMP"] = r"E:\FORGE\.tmp"
os.environ["TORCH_EXTENSIONS_DIR"] = r"E:\FORGE\.tmp\torch_ext"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"E:\FORGE\.tmp\inductor_cache"
os.makedirs(r"E:\FORGE\.tmp", exist_ok=True)

# MSVC environment for TorchInductor on Windows
msvc_bin = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
msvc_inc = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\include"
msvc_lib = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\lib\x64"
ucrt_inc = r"C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\ucrt"
shared_inc = r"C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\shared"
um_inc = r"C:\Program Files (x86)\Windows Kits\10\include\10.0.22621.0\um"
ucrt_lib = r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\ucrt\x64"
um_lib = r"C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\um\x64"

if os.path.exists(msvc_bin) and msvc_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{msvc_bin};" + os.environ.get("PATH", "")
if os.path.exists(msvc_inc):
    current_inc = os.environ.get("INCLUDE", "")
    all_incs = f"{msvc_inc};{ucrt_inc};{shared_inc};{um_inc}"
    if msvc_inc not in current_inc:
        os.environ["INCLUDE"] = f"{all_incs};" + current_inc
if os.path.exists(msvc_lib):
    current_lib = os.environ.get("LIB", "")
    all_libs = f"{msvc_lib};{ucrt_lib};{um_lib}"
    if msvc_lib not in current_lib:
        os.environ["LIB"] = f"{all_libs};" + current_lib

PYTHON_EXE = sys.executable


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def run_benchmark_subprocess(
    backend: str,
    layout: str = "NHWC",
    threads: int = 10,
    interop: int = 1,
    epochs: int = 1,
    warmup_steps: int = 5,
    timed_steps: Optional[int] = None,
    output_json: Optional[str] = None,
    skip_eval: bool = False,
) -> Dict:
    """Launch benchmark in a fresh isolated process."""
    cmd = [
        PYTHON_EXE,
        "-m",
        "forge.benchmarking.canonical_runner",
        "--worker",
        "--backend", backend,
        "--layout", layout,
        "--threads", str(threads),
        "--interop", str(interop),
        "--epochs", str(epochs),
        "--warmup-steps", str(warmup_steps),
    ]
    if timed_steps is not None:
        cmd.extend(["--timed-steps", str(timed_steps)])
    if skip_eval:
        cmd.append("--skip-eval")
    if output_json:
        cmd.extend(["--output-json", output_json])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    try:
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env, timeout=180)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"Benchmark worker timed out after 180 seconds for {backend} ({layout}, T{threads})") from e
    if res.returncode != 0:
        raise RuntimeError(f"Benchmark worker failed with exit code {res.returncode}:\n{res.stderr}\n{res.stdout}")

    if output_json and Path(output_json).exists():
        with open(output_json, "r") as f:
            return json.load(f)
    return {"raw_stdout": res.stdout}


def worker_main():
    """Worker entry point running inside the dedicated subprocess."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--backend", type=str, required=True, choices=["torch-eager", "torch-tuned", "torch-compiled", "forge"])
    parser.add_argument("--layout", type=str, default="NHWC", choices=["NCHW", "NHWC"])
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--interop", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=None)
    parser.add_argument("--skip-eval", action="store_true", default=False)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    # Apply thread settings before any torch ops
    import torch
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.interop)
    except RuntimeError:
        pass

    import torch.nn as nn
    import numpy as np
    from workloads.models.resnet20 import resnet20
    from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders, evaluate

    set_seeds(42)
    device = torch.device("cpu")
    model = resnet20().to(device)

    is_channels_last = (args.layout == "NHWC")
    if is_channels_last:
        model = model.to(memory_format=torch.channels_last)

    compile_overhead_s = 0.0
    if args.backend == "torch-compiled":
        t_comp_start = time.perf_counter()
        model = torch.compile(model, backend="inductor", mode="default")
        # Warmup dummy pass for compilation
        dummy_in = torch.randn(128, 3, 32, 32)
        if is_channels_last:
            dummy_in = dummy_in.to(memory_format=torch.channels_last)
        dummy_out = model(dummy_in)
        dummy_loss = nn.CrossEntropyLoss()(dummy_out, torch.zeros(128, dtype=torch.long))
        dummy_loss.backward()
        del dummy_in, dummy_out, dummy_loss
        for p in model.parameters():
            p.grad = None
        gc.collect()
        compile_overhead_s = time.perf_counter() - t_comp_start

    elif args.backend == "forge":
        from forge.integration.forge_trainer import apply_forge_backend
        model = apply_forge_backend(model)
        if is_channels_last:
            model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=2)

    epoch_records = []
    total_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        step_times = []
        fwd_times = []
        bwd_times = []
        opt_times = []
        data_times = []

        correct = 0
        total = 0
        total_loss = 0.0
        step_idx = 0

        t_epoch_start = time.perf_counter()
        t_data_start = time.perf_counter()

        for inputs, targets in train_loader:
            t_data_end = time.perf_counter()
            data_load_time = t_data_end - t_data_start

            if is_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            t0 = time.perf_counter()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            t1 = time.perf_counter()

            loss.backward()
            t2 = time.perf_counter()

            optimizer.step()
            t3 = time.perf_counter()

            step_duration = t3 - t_data_start
            fwd_duration = t1 - t0
            bwd_duration = t2 - t1
            opt_duration = t3 - t2

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)

            if epoch > 1 or step_idx >= args.warmup_steps:
                step_times.append(step_duration * 1000.0)
                fwd_times.append(fwd_duration * 1000.0)
                bwd_times.append(bwd_duration * 1000.0)
                opt_times.append(opt_duration * 1000.0)
                data_times.append(data_load_time * 1000.0)

            step_idx += 1
            if args.timed_steps is not None and len(step_times) >= args.timed_steps:
                break
            t_data_start = time.perf_counter()

        t_epoch_end = time.perf_counter()

        # Evaluation
        if not args.skip_eval:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        else:
            test_loss, test_acc = None, None

        rec = {
            "epoch": epoch,
            "epoch_time_s": round(t_epoch_end - t_epoch_start, 4),
            "median_step_ms": round(float(np.median(step_times)), 2),
            "mean_step_ms": round(float(np.mean(step_times)), 2),
            "p95_step_ms": round(float(np.percentile(step_times, 95)), 2),
            "median_fwd_ms": round(float(np.median(fwd_times)), 2),
            "median_bwd_ms": round(float(np.median(bwd_times)), 2),
            "median_opt_ms": round(float(np.median(opt_times)), 2),
            "median_data_ms": round(float(np.median(data_times)), 2),
            "samples_per_sec": round(128.0 / (float(np.median(step_times)) / 1000.0), 1),
            "train_acc_pct": round(correct / total * 100.0, 2),
            "test_acc_pct": round(test_acc, 2) if test_acc is not None else None,
            "test_loss": round(test_loss, 4) if test_loss is not None else None,
            "steps_timed": len(step_times),
        }
        epoch_records.append(rec)

    total_elapsed_s = time.perf_counter() - total_start
    import psutil
    process = psutil.Process()
    peak_rss_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    output_data = {
        "backend": args.backend,
        "layout": args.layout,
        "threads": args.threads,
        "interop": args.interop,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "compile_overhead_s": round(compile_overhead_s, 3),
        "total_elapsed_s": round(total_elapsed_s, 4),
        "peak_rss_mb": peak_rss_mb,
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "cpu_model": platform.processor(),
        "epochs_data": epoch_records,
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)

    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
    else:
        # CLI runner
        print("FORGE Canonical Benchmark Runner")
