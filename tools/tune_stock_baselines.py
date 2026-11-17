"""Systematic Baseline Tuning and Certification for FORGE Milestone M3.
Evaluates:
- Eager NCHW vs NHWC
- Thread sweeps (4, 6, 8, 10, 12, 14, 16)
- Interop threads (1, 2)
- TorchInductor compile configurations (default, reduce-overhead)
"""
import os
import sys
from pathlib import Path
import time
import json
import csv
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure temp directory on Drive E:
os.environ["TMP"] = r"E:\FORGE\.tmp"
os.environ["TEMP"] = r"E:\FORGE\.tmp"
os.environ["TORCH_EXTENSIONS_DIR"] = r"E:\FORGE\.tmp\torch_ext"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"E:\FORGE\.tmp\inductor_cache"
os.makedirs(r"E:\FORGE\.tmp", exist_ok=True)

import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_configuration(
    name: str,
    threads: int,
    interop: int,
    channels_last: bool,
    compiled: bool,
    compile_mode: str = "default",
    num_workers: int = 2,
    warmup_steps: int = 5,
    timed_steps: int = 25,
):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        pass  # cannot change if already initialized

    set_seeds(42)
    device = torch.device("cpu")
    model = resnet20().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    compile_time_s = 0.0
    if compiled:
        t0 = time.perf_counter()
        model = torch.compile(model, backend="inductor", mode=compile_mode)
        # compile warmup
        dummy_x = torch.randn(128, 3, 32, 32)
        if channels_last:
            dummy_x = dummy_x.to(memory_format=torch.channels_last)
        dummy_y = torch.randint(0, 10, (128,))
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        out = model(dummy_x)
        l = crit(out, dummy_y)
        l.backward()
        opt.step()
        compile_time_s = time.perf_counter() - t0

    set_seeds(42)
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    model.train()
    loader_iter = iter(train_loader)
    
    # Warmup
    for _ in range(warmup_steps):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Timed steps
    step_times = []
    fwd_times = []
    bwd_times = []
    opt_times = []

    for _ in range(timed_steps):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        t_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        
        t0 = time.perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        t1 = time.perf_counter()
        
        loss.backward()
        t2 = time.perf_counter()
        
        optimizer.step()
        t3 = time.perf_counter()

        step_times.append((t3 - t_start) * 1000.0)
        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)
        opt_times.append((t3 - t2) * 1000.0)

    import numpy as np
    median_step = float(np.median(step_times))
    mean_step = float(np.mean(step_times))
    p95_step = float(np.percentile(step_times, 95))
    throughput = 128.0 / (median_step / 1000.0)

    result = {
        "name": name,
        "threads": threads,
        "interop": interop,
        "layout": "NHWC" if channels_last else "NCHW",
        "compiled": compiled,
        "compile_mode": compile_mode if compiled else "none",
        "compile_time_s": round(compile_time_s, 2),
        "median_step_ms": round(median_step, 2),
        "mean_step_ms": round(mean_step, 2),
        "p95_step_ms": round(p95_step, 2),
        "median_fwd_ms": round(float(np.median(fwd_times)), 2),
        "median_bwd_ms": round(float(np.median(bwd_times)), 2),
        "median_opt_ms": round(float(np.median(opt_times)), 2),
        "samples_per_sec": round(throughput, 1),
    }
    print(f"[{name:<24}] median={median_step:>6.2f}ms | fwd={result['median_fwd_ms']:>5.2f}ms | bwd={result['median_bwd_ms']:>5.2f}ms | smp/s={throughput:>6.1f}")
    return result


def main():
    print("=" * 80)
    print("FORGE M3: SYSTEMATIC BASELINE RE-EVALUATION & TUNING SWEEP")
    print("=" * 80)
    
    results = []
    
    # 1. Thread sweeps on NHWC (Channels-Last)
    for t in [4, 6, 8, 10, 12, 16]:
        res = run_configuration(
            name=f"Stock_NHWC_T{t}",
            threads=t,
            interop=1,
            channels_last=True,
            compiled=False,
        )
        results.append(res)

    # 2. Layout comparisons on NCHW (Contiguous)
    for t in [8, 10, 12]:
        res = run_configuration(
            name=f"Stock_NCHW_T{t}",
            threads=t,
            interop=1,
            channels_last=False,
            compiled=False,
        )
        results.append(res)

    # 3. TorchInductor compiled
    for mode in ["default", "reduce-overhead"]:
        res = run_configuration(
            name=f"Compiled_NHWC_{mode}",
            threads=10,
            interop=1,
            channels_last=True,
            compiled=True,
            compile_mode=mode,
            timed_steps=15,
        )
        results.append(res)

    out_csv = PROJECT_ROOT / "audit" / "m3" / "baseline_tuning_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {out_csv}")

if __name__ == "__main__":
    main()
