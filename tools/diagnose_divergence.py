"""Paired step-by-step divergence diagnostic between PyTorch eager and FORGE backend."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import random
import numpy as np
import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders
from forge.integration.forge_trainer import apply_forge_backend

def diagnose():
    print("=== FORGE vs EAGER STEP-BY-STEP DIVERGENCE DIAGNOSTIC ===")
    
    # 1. Initialize exact identical weights
    set_seeds(42)
    model_eager = resnet20()
    
    set_seeds(42)
    model_forge = resnet20()
    
    # Verify weights match at init
    for (n1, p1), (n2, p2) in zip(model_eager.named_parameters(), model_forge.named_parameters()):
        diff = (p1 - p2).abs().max().item()
        assert diff == 0.0, f"Initial weight mismatch in {n1}: {diff}"
    print("Initial weights strictly identical.")
    
    # Wrap model_forge
    model_forge = apply_forge_backend(model_forge)
    
    criterion = nn.CrossEntropyLoss()
    opt_eager = torch.optim.SGD(model_eager.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    opt_forge = torch.optim.SGD(model_forge.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Deterministic fixed batch
    torch.manual_seed(123)
    x = torch.randn(128, 3, 32, 32)
    y = torch.randint(0, 10, (128,))
    
    for step in range(5):
        model_eager.train()
        model_forge.train()
        
        opt_eager.zero_grad()
        opt_forge.zero_grad()
        
        out_eager = model_eager(x)
        out_forge = model_forge(x)
        
        fwd_diff = (out_eager - out_forge).abs().max().item()
        
        loss_eager = criterion(out_eager, y)
        loss_forge = criterion(out_forge, y)
        loss_diff = abs(loss_eager.item() - loss_forge.item())
        
        loss_eager.backward()
        loss_forge.backward()
        
        # Check gradients
        max_grad_diff = 0.0
        max_grad_param = ""
        for (n1, p1), (n2, p2) in zip(model_eager.named_parameters(), model_forge.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                gdiff = (p1.grad - p2.grad).abs().max().item()
                if gdiff > max_grad_diff:
                    max_grad_diff = gdiff
                    max_grad_param = n1
        
        opt_eager.step()
        opt_forge.step()
        
        # Check weights after step
        max_weight_diff = 0.0
        for (n1, p1), (n2, p2) in zip(model_eager.named_parameters(), model_forge.named_parameters()):
            wdiff = (p1 - p2).abs().max().item()
            if wdiff > max_weight_diff:
                max_weight_diff = wdiff
                
        # Check BN running stats
        max_bn_diff = 0.0
        for (n1, b1), (n2, b2) in zip(model_eager.named_buffers(), model_forge.named_buffers()):
            bdiff = (b1 - b2).abs().max().item()
            if bdiff > max_bn_diff:
                max_bn_diff = bdiff
        
        print(f"Step {step+1}: Fwd diff={fwd_diff:.2e}, Loss diff={loss_diff:.2e}, "
              f"Max grad diff={max_grad_diff:.2e} ({max_grad_param}), "
              f"Weight diff={max_weight_diff:.2e}, BN running diff={max_bn_diff:.2e}")

if __name__ == "__main__":
    diagnose()
