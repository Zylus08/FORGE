"""Profile the compiled baseline (B2: torch.compile) on ResNet-20 CPU."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import json
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders

def profile_compiled():
    torch.set_num_threads(10)
    torch.set_num_interop_threads(1)
    device = torch.device("cpu")
    
    set_seeds(42)
    model = resnet20().to(device)
    model = model.to(memory_format=torch.channels_last)
    
    print("Compiling model with torch.compile...")
    compiled_model = torch.compile(model, backend="inductor", mode="default")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(compiled_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Warmup dummy steps to trigger compilation before profiling
    dummy_x = torch.randn(128, 3, 32, 32).to(memory_format=torch.channels_last)
    dummy_y = torch.randint(0, 10, (128,))
    for _ in range(3):
        out = compiled_model(dummy_x)
        l = criterion(out, dummy_y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("Warmup complete. Starting profiler...")
    
    set_seeds(42)
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=2)
    loader_iter = iter(train_loader)
    
    trace_path = "benchmarks/results/profile_compiled.json"
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    
    compiled_model.train()
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(10):
            with record_function(f"## step_{step} ##"):
                inputs, targets = next(loader_iter)
                inputs = inputs.to(memory_format=torch.channels_last)
                
                optimizer.zero_grad()
                outputs = compiled_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
    print("\nProfiler completed. Exporting Chrome trace...")
    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to {trace_path}")
    
    key_averages = prof.key_averages()
    print("\n" + "=" * 90)
    print("TOP 25 OPERATORS BY SELF CPU TIME (COMPILED BASELINE B2)")
    print("=" * 90)
    print(key_averages.table(sort_by="self_cpu_time_total", row_limit=25))
    
    # Analyze breakdown
    conv_time = 0.0
    bn_time = 0.0
    relu_time = 0.0
    inductor_cpp_time = 0.0
    opt_time = 0.0
    total_time = 0.0
    
    for evt in key_averages:
        t = evt.self_cpu_time_total
        name = evt.key
        total_time += t
        if "convolution" in name or "mkldnn" in name:
            conv_time += t
        elif "batch_norm" in name:
            bn_time += t
        elif "relu" in name or "clamp" in name or "threshold" in name:
            relu_time += t
        elif "cpp_fused" in name or "inductor" in name:
            inductor_cpp_time += t
        elif "sgd" in name or "optimizer" in name:
            opt_time += t

    print("\n" + "=" * 60)
    print("COMPILED BASELINE OPERATOR BREAKDOWN")
    print("=" * 60)
    print(f"Total Self CPU Time : {total_time/1000.0:8.2f} ms")
    print(f"Convolution         : {conv_time/1000.0:8.2f} ms ({conv_time/total_time*100:5.1f}%)")
    print(f"Batch Normalization : {bn_time/1000.0:8.2f} ms ({bn_time/total_time*100:5.1f}%)")
    print(f"Inductor C++ Fused  : {inductor_cpp_time/1000.0:8.2f} ms ({inductor_cpp_time/total_time*100:5.1f}%)")
    print(f"ReLU / Activations  : {relu_time/1000.0:8.2f} ms ({relu_time/total_time*100:5.1f}%)")
    print(f"Optimizer           : {opt_time/1000.0:8.2f} ms ({opt_time/total_time*100:5.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    profile_compiled()
