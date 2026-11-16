#!/usr/bin/env python3
"""FORGE M1 Baseline Benchmark Runner.

Usage:
  python -m forge.benchmarking.run_baselines --backend torch-eager --epochs 5
  python -m forge.benchmarking.run_baselines --backend torch-tuned --epochs 5
  python -m forge.benchmarking.run_baselines --backend torch-compiled --epochs 5
  python -m forge.benchmarking.run_baselines --backend forge --epochs 5
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from workloads.models.resnet20 import resnet20

# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = False,
    data_root: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Return (train_loader, test_loader) for CIFAR-10."""
    if data_root is None:
        if (Path("E:/FORGE/data") / "cifar-10-batches-py").exists() or Path("E:/FORGE/data").exists():
            data_root = "E:/FORGE/data"
        else:
            data_root = str(PROJECT_ROOT / "workloads" / "datasets")
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

def configure_backend(backend: str, model: nn.Module) -> nn.Module:
    """Apply backend-specific settings and return the (possibly compiled) model."""
    physical_cores = psutil.cpu_count(logical=False) or 1

    if backend == "torch-eager":
        # B0: pure defaults — do nothing
        pass

    elif backend == "torch-tuned":
        # B1: thread tuning + channels_last
        torch.set_num_threads(physical_cores)
        torch.set_num_interop_threads(1)
        model = model.to(memory_format=torch.channels_last)

    elif backend == "torch-compiled":
        # B2: torch.compile with inductor
        torch.set_num_threads(physical_cores)
        torch.set_num_interop_threads(1)
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")

    elif backend == "forge":
        # Forge backend: attempt to load native extension; fall back to tuned
        torch.set_num_threads(physical_cores)
        torch.set_num_interop_threads(1)
        try:
            from forge.integration.forge_trainer import apply_forge_backend
            model = apply_forge_backend(model)
            print("[forge] Native backend applied.")
        except Exception as exc:
            print(f"[forge] Native backend unavailable ({exc}); falling back to torch-tuned.")
            model = model.to(memory_format=torch.channels_last)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    return model


# ---------------------------------------------------------------------------
# Per-step timed training
# ---------------------------------------------------------------------------

def timed_train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    warmup_steps: int = 2,
    backend: str = "torch-eager",
) -> Dict:
    """Train one epoch; return dict of timing arrays and accuracy."""
    model.train()

    step_times: List[float] = []
    data_times: List[float] = []
    fwd_times: List[float] = []
    bwd_times: List[float] = []
    opt_times: List[float] = []

    correct = 0
    total = 0
    step_idx = 0

    t_epoch_start = time.perf_counter()
    t_data_start = time.perf_counter()

    for inputs, targets in loader:
        t_data_end = time.perf_counter()
        data_load = t_data_end - t_data_start

        # Apply channels_last if needed
        if backend == "torch-tuned":
            inputs = inputs.to(memory_format=torch.channels_last)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Forward
        t_fwd_start = time.perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        t_fwd_end = time.perf_counter()

        # Backward
        optimizer.zero_grad()
        t_bwd_start = time.perf_counter()
        loss.backward()
        t_bwd_end = time.perf_counter()

        # Optimizer step
        t_opt_start = time.perf_counter()
        optimizer.step()
        t_opt_end = time.perf_counter()

        step_total = (t_opt_end - t_data_start)

        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        # Only record after warmup
        if step_idx >= warmup_steps:
            step_times.append(step_total)
            data_times.append(data_load)
            fwd_times.append(t_fwd_end - t_fwd_start)
            bwd_times.append(t_bwd_end - t_bwd_start)
            opt_times.append(t_opt_end - t_opt_start)

        step_idx += 1
        t_data_start = time.perf_counter()

    epoch_time = time.perf_counter() - t_epoch_start
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    batch_size_actual = loader.batch_size or 128
    samples_per_sec = (len(loader.dataset) / epoch_time) if epoch_time > 0 else 0.0

    return {
        "epoch_time_s": epoch_time,
        "samples_per_sec": samples_per_sec,
        "accuracy_pct": accuracy,
        "step_times": step_times,
        "data_times": data_times,
        "fwd_times": fwd_times,
        "bwd_times": bwd_times,
        "opt_times": opt_times,
        "num_steps": len(step_times),
    }


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (test_loss, test_accuracy_pct)."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(args) -> List[Dict]:
    """Run the full benchmark for the given args; return list of epoch result dicts."""
    set_seeds(args.seed)

    device = torch.device("cpu")
    physical_cores = psutil.cpu_count(logical=False) or 1
    print(f"\n{'='*60}")
    print(f"FORGE M1 Baseline Benchmark")
    print(f"Backend : {args.backend}")
    print(f"Epochs  : {args.epochs}")
    print(f"Seed    : {args.seed}")
    print(f"Physical cores: {physical_cores}")
    print(f"{'='*60}\n")

    # Num workers for data loading
    num_workers = 2 if args.backend in ("torch-tuned", "torch-compiled", "forge") else 0

    # Build model (pre-compile version for timing if compiled)
    model_base = resnet20().to(device)

    # Measure compilation time for torch-compiled backend
    compilation_time_s: Optional[float] = None
    if args.backend == "torch-compiled":
        print("Measuring compilation overhead...")
        # Warm up with a fake forward/backward to trigger compilation
        t_compile_start = time.perf_counter()
        model_compiled = configure_backend(args.backend, resnet20().to(device))
        # Trigger compilation with a dummy forward+backward
        dummy_input = torch.randn(128, 3, 32, 32)
        dummy_target = torch.zeros(128, dtype=torch.long)
        criterion_dummy = nn.CrossEntropyLoss()
        optimizer_dummy = torch.optim.SGD(model_compiled.parameters(), lr=0.01)
        try:
            out = model_compiled(dummy_input)
            loss = criterion_dummy(out, dummy_target)
            loss.backward()
            optimizer_dummy.step()
        except Exception as e:
            print(f"  [warn] Compilation warm-up error: {e}")
        t_compile_end = time.perf_counter()
        compilation_time_s = t_compile_end - t_compile_start
        print(f"  Compilation time: {compilation_time_s:.2f}s")
        model = model_compiled
        set_seeds(args.seed)
    else:
        model = configure_backend(args.backend, model_base)

    # Data loaders
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=128,
        num_workers=num_workers,
        pin_memory=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 120], gamma=0.1
    )

    epoch_results: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_metrics = timed_train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            warmup_steps=args.warmup_steps if epoch == 1 else 0,
            backend=args.backend,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        st = train_metrics["step_times"]
        median_step = float(np.median(st)) if st else 0.0
        p95_step = float(np.percentile(st, 95)) if st else 0.0

        result = {
            "backend": args.backend,
            "epoch": epoch,
            "epoch_time_s": round(train_metrics["epoch_time_s"], 4),
            "samples_per_sec": round(train_metrics["samples_per_sec"], 2),
            "train_accuracy_pct": round(train_metrics["accuracy_pct"], 4),
            "test_accuracy_pct": round(test_acc, 4),
            "test_loss": round(test_loss, 6),
            "median_step_time_ms": round(median_step * 1000, 3),
            "p95_step_time_ms": round(p95_step * 1000, 3),
            "median_data_time_ms": round(float(np.median(train_metrics["data_times"])) * 1000, 3) if train_metrics["data_times"] else 0.0,
            "median_fwd_time_ms": round(float(np.median(train_metrics["fwd_times"])) * 1000, 3) if train_metrics["fwd_times"] else 0.0,
            "median_bwd_time_ms": round(float(np.median(train_metrics["bwd_times"])) * 1000, 3) if train_metrics["bwd_times"] else 0.0,
            "median_opt_time_ms": round(float(np.median(train_metrics["opt_times"])) * 1000, 3) if train_metrics["opt_times"] else 0.0,
            "num_steps_timed": train_metrics["num_steps"],
            "compilation_time_s": round(compilation_time_s, 3) if compilation_time_s is not None else None,
        }
        epoch_results.append(result)

        print(
            f"  train_acc={result['train_accuracy_pct']:.2f}%  "
            f"test_acc={result['test_accuracy_pct']:.2f}%  "
            f"epoch_time={result['epoch_time_s']:.2f}s  "
            f"samples/s={result['samples_per_sec']:.1f}  "
            f"median_step={result['median_step_time_ms']:.1f}ms  "
            f"[fwd={result['median_fwd_time_ms']:.1f}  "
            f"bwd={result['median_bwd_time_ms']:.1f}  "
            f"opt={result['median_opt_time_ms']:.1f}  "
            f"data={result['median_data_time_ms']:.1f}]ms"
        )

    return epoch_results


def print_summary_table(results: List[Dict]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'='*100}")
    print(f"{'FORGE M1 BENCHMARK SUMMARY':^100}")
    print(f"{'='*100}")
    header = (
        f"{'Backend':<18} {'Ep':>3} {'Time(s)':>8} {'Smp/s':>8} "
        f"{'Trn%':>6} {'Tst%':>6} {'StepMs':>8} "
        f"{'FwdMs':>7} {'BwdMs':>7} {'OptMs':>7} {'DataMs':>7}"
    )
    print(header)
    print("-" * 100)
    for r in results:
        line = (
            f"{r['backend']:<18} {r['epoch']:>3} {r['epoch_time_s']:>8.2f} "
            f"{r['samples_per_sec']:>8.1f} {r['train_accuracy_pct']:>6.2f} "
            f"{r['test_accuracy_pct']:>6.2f} {r['median_step_time_ms']:>8.1f} "
            f"{r['median_fwd_time_ms']:>7.1f} {r['median_bwd_time_ms']:>7.1f} "
            f"{r['median_opt_time_ms']:>7.1f} {r['median_data_time_ms']:>7.1f}"
        )
        print(line)
    print("=" * 100)


def save_csv(results: List[Dict], path: str) -> None:
    """Save results to CSV."""
    if not results:
        return
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {out_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FORGE M1 Baseline Benchmark Runner")
    parser.add_argument(
        "--backend",
        choices=["torch-eager", "torch-tuned", "torch-compiled", "forge"],
        default="torch-eager",
        help="Training backend to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to write CSV results")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warm-up steps before timing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = run_benchmark(args)
    print_summary_table(results)
    if args.output_csv:
        save_csv(results, args.output_csv)
    # Return last epoch test accuracy for quick inspection
    if results:
        last = results[-1]
        print(f"\nFinal test accuracy: {last['test_accuracy_pct']:.2f}%")
    return results


if __name__ == "__main__":
    main()
