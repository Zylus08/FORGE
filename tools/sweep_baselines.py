"""Baseline Reinforcement & Tuning Sweep for PyTorch CPU.
Sweeps:
- Intra-op thread count (4, 6, 8, 10, 12, 16)
- Inter-op thread count (1, 2)
- Memory format (contiguous NCHW vs channels_last NHWC)
- torch.compile backends & modes
- Data loader workers (0, 2)
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import psutil
import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders

def benchmark_config(
    name: str,
    threads: int,
    interop: int,
    channels_last: bool,
    compiled: bool,
    compile_mode: str = "default",
    num_workers: int = 0,
    steps: int = 30,
    warmup: int = 5,
):
    try:
        torch.set_num_threads(threads)
    except Exception as e:
        pass

    device = torch.device("cpu")
    set_seeds(42)
    model = resnet20().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    compile_time = 0.0
    if compiled:
        t0 = time.perf_counter()
        # Compile fresh model
        model = torch.compile(model, backend="inductor", mode=compile_mode)
        # Warmup dummy pass for compilation
        dummy_x = torch.randn(128, 3, 32, 32)
        if channels_last:
            dummy_x = dummy_x.to(memory_format=torch.channels_last)
        dummy_y = torch.randint(0, 10, (128,))
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        out = model(dummy_x)
        l = crit(out, dummy_y)
        l.backward()
        compile_time = time.perf_counter() - t0

    set_seeds(42)
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    model.train()
    step_times = []
    
    loader_iter = iter(train_loader)
    for i in range(warmup + steps):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            inputs, targets = next(loader_iter)

        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        t_start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        t_end = time.perf_counter()

        if i >= warmup:
            step_times.append((t_end - t_start) * 1000.0)

    median_ms = sorted(step_times)[len(step_times) // 2]
    mean_ms = sum(step_times) / len(step_times)
    sps = 128.0 / (median_ms / 1000.0)
    print(f"[{name:<28}] Threads={threads:2d} | CH_Last={str(channels_last):<5} | "
          f"Compiled={str(compiled):<5} | Mode={compile_mode:<15} | "
          f"Median={median_ms:6.1f} ms | Mean={mean_ms:6.1f} ms | {sps:6.1f} smp/s | Compile={compile_time:5.1f}s")
    return {
        "name": name,
        "threads": threads,
        "channels_last": channels_last,
        "compiled": compiled,
        "compile_mode": compile_mode,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "samples_per_sec": sps,
        "compile_time_s": compile_time,
    }

def main():
    print("=" * 110)
    print("FORGE M2: BASELINE REINFORCEMENT & SYSTEMATIC CPU TUNING SWEEP")
    print("=" * 110)
    
    results = []

    # 1. Thread Sweep (Eager, Contiguous)
    print("\n--- 1. Intra-op Thread Sweep (Eager, NCHW Contiguous) ---")
    for th in [4, 6, 8, 10, 12, 16]:
        r = benchmark_config(f"Eager_T{th}_Contig", threads=th, interop=1, channels_last=False, compiled=False)
        results.append(r)

    # 2. Memory Format Sweep (Channels-Last NHWC)
    print("\n--- 2. Memory Format Sweep (Eager, Channels-Last NHWC) ---")
    for th in [6, 8, 10]:
        r = benchmark_config(f"Eager_T{th}_NHWC", threads=th, interop=1, channels_last=True, compiled=False)
        results.append(r)

    # 3. Torch.compile Sweep
    print("\n--- 3. Torch.compile Sweeps (Inductor, Modes: default, reduce-overhead) ---")
    for th in [6, 8, 10]:
        r = benchmark_config(f"Compile_T{th}_default", threads=th, interop=1, channels_last=False, compiled=True, compile_mode="default")
        results.append(r)
        r = benchmark_config(f"Compile_T{th}_NHWC_default", threads=th, interop=1, channels_last=True, compiled=True, compile_mode="default")
        results.append(r)

    print("\n" + "=" * 110)
    print("SUMMARY: RANKED BY MEDIAN STEP TIME")
    print("=" * 110)
    results.sort(key=lambda x: x["median_ms"])
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['name']:<28} : {r['median_ms']:6.1f} ms/step  ({r['samples_per_sec']:6.1f} samples/s)")

if __name__ == "__main__":
    main()
