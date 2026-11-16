"""Controlled Ablation Study for FORGE M2 Optimization Pipeline.
Evaluates:
1. B0: Stock Eager (NCHW, 10 threads, un-tuned)
2. B1: Stock Tuned (NHWC, 10 threads, oneDNN/MKL)
3. B2: Stock Compiled (torch.compile, 10 threads)
4. M1 FORGE Prototype: Python autograd.Function wrapper + Fused ReLU (NCHW)
5. M2 Dispatch Optimization: C++ autograd dispatch + Fused ReLU (NCHW)
6. M2 Fused Residual: C++ autograd dispatch + Fused Add+ReLU (NCHW)
7. M2 Full Optimized: C++ autograd dispatch + Fused Add+ReLU + Channels-Last (NHWC)
"""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import csv
import torch
# Set thread counts at module load before any parallel work starts
try:
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
except Exception:
    pass
import torch.nn as nn
from workloads.models.resnet20 import resnet20, BasicBlock
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders
from forge.backends.forge_backend import (
    FusedReLUFunction,
    fused_relu_cpp,
    fused_add_relu_cpp,
)

# Custom model with Fused Add+ReLU in BasicBlock
class ForgeOptimizedBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A', use_cpp_autograd=True, use_fused_add_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_cpp_autograd = use_cpp_autograd
        self.use_fused_add_relu = use_fused_add_relu
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                from workloads.models.resnet20 import LambdaLayer
                import torch.nn.functional as F
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)
                )

    def forward(self, x):
        if self.use_cpp_autograd:
            out = fused_relu_cpp(self.bn1(self.conv1(x)))
        else:
            out = FusedReLUFunction.apply(self.bn1(self.conv1(x)))
            
        out = self.bn2(self.conv2(out))
        res = self.shortcut(x)
        
        if self.use_fused_add_relu:
            out = fused_add_relu_cpp(out, res)
        else:
            out = out + res
            if self.use_cpp_autograd:
                out = fused_relu_cpp(out)
            else:
                out = FusedReLUFunction.apply(out)
        return out


def make_forge_model(use_cpp_autograd=True, use_fused_add_relu=True):
    set_seeds(42)
    model = resnet20()
    # Replace blocks with ForgeOptimizedBlock
    for layer_name in ["layer1", "layer2", "layer3"]:
        layer = getattr(model, layer_name)
        new_blocks = []
        for block in layer:
            in_p = block.conv1.in_channels
            p = block.conv1.out_channels
            s = block.conv1.stride[0]
            nb = ForgeOptimizedBlock(in_p, p, stride=s, use_cpp_autograd=use_cpp_autograd, use_fused_add_relu=use_fused_add_relu)
            # Copy trained weights
            nb.conv1.weight.data.copy_(block.conv1.weight.data)
            nb.bn1.weight.data.copy_(block.bn1.weight.data)
            nb.bn1.bias.data.copy_(block.bn1.bias.data)
            nb.bn1.running_mean.copy_(block.bn1.running_mean)
            nb.bn1.running_var.copy_(block.bn1.running_var)
            nb.conv2.weight.data.copy_(block.conv2.weight.data)
            nb.bn2.weight.data.copy_(block.bn2.weight.data)
            nb.bn2.bias.data.copy_(block.bn2.bias.data)
            nb.bn2.running_mean.copy_(block.bn2.running_mean)
            nb.bn2.running_var.copy_(block.bn2.running_var)
            new_blocks.append(nb)
        setattr(model, layer_name, nn.Sequential(*new_blocks))
    return model


def run_single_ablation(name: str, model: nn.Module, channels_last: bool, warmup: int = 5, steps: int = 30):
    device = torch.device("cpu")
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        
    set_seeds(42)
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    model.train()
    step_times = []
    fwd_times = []
    bwd_times = []

    loader_iter = iter(train_loader)
    for i in range(warmup + steps):
        inputs, targets = next(loader_iter)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)

        t_start = time.perf_counter()
        optimizer.zero_grad()
        
        t_fwd_0 = time.perf_counter()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        t_fwd_1 = time.perf_counter()
        
        t_bwd_0 = time.perf_counter()
        loss.backward()
        t_bwd_1 = time.perf_counter()
        
        optimizer.step()
        t_end = time.perf_counter()

        if i >= warmup:
            step_times.append((t_end - t_start) * 1000.0)
            fwd_times.append((t_fwd_1 - t_fwd_0) * 1000.0)
            bwd_times.append((t_bwd_1 - t_bwd_0) * 1000.0)

    median_step = sorted(step_times)[len(step_times) // 2]
    median_fwd = sorted(fwd_times)[len(fwd_times) // 2]
    median_bwd = sorted(bwd_times)[len(bwd_times) // 2]
    sps = 128.0 / (median_step / 1000.0)

    print(f"[{name:<32}] Step: {median_step:6.1f} ms | Fwd: {median_fwd:5.1f} ms | Bwd: {median_bwd:5.1f} ms | {sps:6.1f} smp/s")
    return {
        "configuration": name,
        "median_step_ms": round(median_step, 2),
        "median_fwd_ms": round(median_fwd, 2),
        "median_bwd_ms": round(median_bwd, 2),
        "samples_per_sec": round(sps, 2),
    }


def main():
    print("=" * 100)
    print("FORGE M2 CONTROLLED ABLATION STUDY")
    print("=" * 100)
    ablations = []

    # 1. B0: Stock Eager NCHW
    set_seeds(42)
    m_b0 = resnet20()
    r = run_single_ablation("1. Stock Eager (B0, NCHW)", m_b0, channels_last=False)
    ablations.append(r)

    # 2. B1: Stock Tuned NHWC
    set_seeds(42)
    m_b1 = resnet20()
    r = run_single_ablation("2. Stock Tuned (B1, NHWC)", m_b1, channels_last=True)
    ablations.append(r)

    # 3. M1 FORGE Prototype: Python autograd + Fused ReLU (NCHW)
    m_m1 = make_forge_model(use_cpp_autograd=False, use_fused_add_relu=False)
    r = run_single_ablation("3. M1 FORGE Prototype (Python autograd)", m_m1, channels_last=False)
    ablations.append(r)

    # 4. M2: C++ Autograd Dispatcher + Fused ReLU (NCHW)
    m_m2_disp = make_forge_model(use_cpp_autograd=True, use_fused_add_relu=False)
    r = run_single_ablation("4. M2 C++ Autograd Dispatch (NCHW)", m_m2_disp, channels_last=False)
    ablations.append(r)

    # 5. M2: C++ Autograd + Fused Add+ReLU (NCHW)
    m_m2_fused = make_forge_model(use_cpp_autograd=True, use_fused_add_relu=True)
    r = run_single_ablation("5. M2 Fused Add+ReLU (NCHW)", m_m2_fused, channels_last=False)
    ablations.append(r)

    # 6. M2 Full Optimized: C++ Autograd + Fused Add+ReLU + Channels-Last (NHWC)
    m_m2_opt = make_forge_model(use_cpp_autograd=True, use_fused_add_relu=True)
    r = run_single_ablation("6. M2 Full Optimized (NHWC)", m_m2_opt, channels_last=True)
    ablations.append(r)

    # Calculate speedups vs B0 and vs M1
    b0_step = ablations[0]["median_step_ms"]
    m1_step = ablations[2]["median_step_ms"]
    for row in ablations:
        row["speedup_vs_B0"] = round(b0_step / row["median_step_ms"], 3)
        row["speedup_vs_M1"] = round(m1_step / row["median_step_ms"], 3)

    out_csv = "audit/m2/ABLATION_RESULTS.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablations[0].keys()))
        writer.writeheader()
        writer.writerows(ablations)
    print(f"\nAblation results written to {out_csv}")

if __name__ == "__main__":
    main()
