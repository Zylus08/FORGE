#!/usr/bin/env python3
"""FORGE M1 Training Profiler.

Profiles 20 steps of eager ResNet-20 training using torch.profiler.
Exports Chrome trace and prints operator table.

Usage:
  python -m forge.profiling.profile_training
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from workloads.models.resnet20 import resnet20

TRACE_OUTPUT = PROJECT_ROOT / "benchmarks" / "results" / "profile_eager.json"
PROFILE_STEPS = 20
BATCH_SIZE = 128
DATA_ROOT = "E:/FORGE/data" if Path("E:/FORGE/data").exists() else str(PROJECT_ROOT / "workloads" / "datasets")


def get_data_loader():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    return loader


def run_profile():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = resnet20().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    loader = get_data_loader()
    loader_iter = iter(loader)

    # Warm up 2 steps outside profiler
    print("Warming up...")
    for _ in range(2):
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inputs, targets = next(loader_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Profiling {PROFILE_STEPS} steps...")
    TRACE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for step_i in range(PROFILE_STEPS):
            try:
                inputs, targets = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                inputs, targets = next(loader_iter)

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.profiler.record_function("data_to_device"):
                pass  # already done above

            with torch.profiler.record_function("forward"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            with torch.profiler.record_function("backward"):
                optimizer.zero_grad()
                loss.backward()

            with torch.profiler.record_function("optimizer_step"):
                optimizer.step()

            prof.step()

    # Export Chrome trace
    print(f"\nExporting Chrome trace to: {TRACE_OUTPUT}")
    try:
        prof.export_chrome_trace(str(TRACE_OUTPUT))
    except Exception as e:
        print(f"Notice during trace export: {e}")

    # Print top-20 operators by self CPU time
    print("\n" + "=" * 90)
    print("TOP-20 OPERATORS BY SELF CPU TIME")
    print("=" * 90)
    key_avgs = prof.key_averages(group_by_input_shape=False)
    sorted_ops = sorted(key_avgs, key=lambda x: x.self_cpu_time_total, reverse=True)

    header = f"{'Operator':<45} {'Count':>6} {'Self CPU(ms)':>14} {'CPU Total(ms)':>14}"
    print(header)
    print("-" * 90)
    for evt in sorted_ops[:20]:
        print(
            f"{evt.key:<45} {evt.count:>6} "
            f"{evt.self_cpu_time_total/1000:>14.3f} "
            f"{evt.cpu_time_total/1000:>14.3f}"
        )

    # Group by operation type
    print("\n" + "=" * 90)
    print("TIME BY LAYER TYPE (ms total across profiled steps)")
    print("=" * 90)

    type_groups = {
        "conv": 0.0,
        "batch_norm": 0.0,
        "relu": 0.0,
        "linear": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "other": 0.0,
    }

    for evt in key_avgs:
        k = evt.key.lower()
        t = evt.self_cpu_time_total / 1000  # ms

        if any(x in k for x in ["convolution", "conv2d", "mkldnn_convolution", "slow_conv"]):
            type_groups["conv"] += t
        elif any(x in k for x in ["batch_norm", "batchnorm", "native_batch_norm"]):
            type_groups["batch_norm"] += t
        elif "relu" in k:
            type_groups["relu"] += t
        elif any(x in k for x in ["linear", "addmm", "mm"]):
            type_groups["linear"] += t
        elif any(x in k for x in ["cross_entropy", "nll_loss", "log_softmax"]):
            type_groups["loss"] += t
        elif any(x in k for x in ["backward", "grad", "autograd"]):
            type_groups["backward"] += t
        else:
            type_groups["other"] += t

    total_measured = sum(type_groups.values())
    print(f"{'Category':<20} {'Time(ms)':>12} {'Pct':>8}")
    print("-" * 45)
    for category, t_ms in sorted(type_groups.items(), key=lambda x: x[1], reverse=True):
        pct = 100.0 * t_ms / total_measured if total_measured > 0 else 0.0
        print(f"{category:<20} {t_ms:>12.3f} {pct:>7.1f}%")
    print(f"{'TOTAL':<20} {total_measured:>12.3f} {'100.0%':>8}")

    # Save findings to markdown
    findings_path = PROJECT_ROOT / "audit" / "m1" / "PROFILER_FINDINGS.md"
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    with open(findings_path, "w") as f:
        f.write("# FORGE M1 Profiler Findings\n\n")
        f.write(f"**Profile:** {PROFILE_STEPS} steps of eager ResNet-20 training (batch=128, CPU)\n\n")
        f.write("## Top-20 Operators by Self CPU Time\n\n")
        f.write("| Operator | Count | Self CPU (ms) | CPU Total (ms) |\n")
        f.write("|----------|-------|--------------|----------------|\n")
        for evt in sorted_ops[:20]:
            f.write(
                f"| {evt.key} | {evt.count} | "
                f"{evt.self_cpu_time_total/1000:.3f} | "
                f"{evt.cpu_time_total/1000:.3f} |\n"
            )

        f.write("\n## Time by Layer Type\n\n")
        f.write("| Category | Time (ms) | % of Total |\n")
        f.write("|----------|-----------|------------|\n")
        for category, t_ms in sorted(type_groups.items(), key=lambda x: x[1], reverse=True):
            pct = 100.0 * t_ms / total_measured if total_measured > 0 else 0.0
            f.write(f"| {category} | {t_ms:.3f} | {pct:.1f}% |\n")

        f.write("\n## Bottleneck Analysis\n\n")
        sorted_groups = sorted(type_groups.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_groups[:3]
        for i, (cat, t_ms) in enumerate(top3, 1):
            pct = 100.0 * t_ms / total_measured if total_measured > 0 else 0.0
            f.write(f"**#{i} Bottleneck: {cat.upper()}** — {t_ms:.1f}ms ({pct:.1f}% of total)\n\n")

        f.write("\n## Amdahl Feasibility Note\n\n")
        # Compute fraction of conv+backward
        p_conv = type_groups["conv"] / total_measured if total_measured > 0 else 0
        p_bwd = type_groups["backward"] / total_measured if total_measured > 0 else 0
        p_combined = p_conv + p_bwd
        for speedup_kernel in [2.0, 5.0, 10.0]:
            # Amdahl: S = 1 / ((1-p) + p/speedup)
            overall = 1.0 / ((1.0 - p_combined) + p_combined / speedup_kernel)
            f.write(f"- If conv+backward is {p_combined*100:.1f}% of total and kernel speedup = {speedup_kernel:.0f}x → overall speedup = {overall:.2f}x\n")

        f.write(f"\n*Chrome trace exported to: benchmarks/results/profile_eager.json*\n")

    print(f"\nFindings written to: {findings_path}")
    return type_groups, sorted_ops[:20]


if __name__ == "__main__":
    run_profile()
