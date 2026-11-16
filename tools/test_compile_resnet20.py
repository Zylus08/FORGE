"""Benchmark torch.compile on ResNet-20 CIFAR-10 under MSVC x64 environment."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders

def test_compile(threads=10, channels_last=True):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    
    device = torch.device("cpu")
    set_seeds(42)
    model = resnet20().to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
        
    print(f"Compiling model (threads={threads}, channels_last={channels_last})...")
    t0 = time.perf_counter()
    compiled_model = torch.compile(model, backend="inductor", mode="default")
    
    # Warmup dummy step
    dummy_x = torch.randn(128, 3, 32, 32)
    if channels_last:
        dummy_x = dummy_x.to(memory_format=torch.channels_last)
    dummy_y = torch.randint(0, 10, (128,))
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(compiled_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    out = compiled_model(dummy_x)
    l = crit(out, dummy_y)
    l.backward()
    opt.step()
    compile_time = time.perf_counter() - t0
    print(f"Compilation finished in {compile_time:.2f}s")
    
    set_seeds(42)
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=2)
    
    compiled_model.train()
    step_times = []
    
    print("Measuring steady-state steps...")
    loader_iter = iter(train_loader)
    for i in range(25):
        inputs, targets = next(loader_iter)
        if channels_last:
            inputs = inputs.to(memory_format=torch.channels_last)
            
        t_start = time.perf_counter()
        opt.zero_grad()
        outputs = compiled_model(inputs)
        loss = crit(outputs, targets)
        loss.backward()
        opt.step()
        t_end = time.perf_counter()
        
        step_ms = (t_end - t_start) * 1000.0
        if i >= 5: # exclude first 5 warmup steps
            step_times.append(step_ms)
        if i in [0, 1, 5, 10, 20]:
            print(f"  Step {i:2d}: {step_ms:6.1f} ms")
            
    median_ms = sorted(step_times)[len(step_times) // 2]
    mean_ms = sum(step_times) / len(step_times)
    sps = 128.0 / (median_ms / 1000.0)
    print(f"\n[RESULTS] Compiled ResNet-20: Median={median_ms:.1f} ms | Mean={mean_ms:.1f} ms | Throughput={sps:.1f} smp/s")

if __name__ == "__main__":
    test_compile()
