"""Compare 1 full epoch step-by-step between torch-eager and forge."""
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds, get_cifar10_loaders, evaluate
from forge.integration.forge_trainer import apply_forge_backend

def run_comparison():
    print("Testing 1 Epoch Comparison: torch-eager vs forge on CIFAR-10")
    
    # Run 1: torch-eager
    set_seeds(42)
    train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=0)
    model_eager = resnet20()
    criterion = nn.CrossEntropyLoss()
    optimizer_eager = torch.optim.SGD(model_eager.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    model_eager.train()
    eager_losses = []
    for step, (inputs, targets) in enumerate(train_loader):
        optimizer_eager.zero_grad()
        outputs = model_eager(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_eager.step()
        eager_losses.append(loss.item())
        if step in [0, 10, 50, 100, 200, 390]:
            print(f"[Eager] Step {step}: loss = {loss.item():.4f}")
    
    test_loss_eager, test_acc_eager = evaluate(model_eager, test_loader, criterion, torch.device("cpu"))
    print(f"[Eager Epoch 1] Test Loss: {test_loss_eager:.4f}, Test Acc: {test_acc_eager:.2f}%")
    
    # Run 2: forge
    set_seeds(42)
    train_loader2, test_loader2 = get_cifar10_loaders(batch_size=128, num_workers=0)
    model_forge = resnet20()
    model_forge = apply_forge_backend(model_forge)
    optimizer_forge = torch.optim.SGD(model_forge.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    model_forge.train()
    forge_losses = []
    max_step_diff = 0.0
    for step, (inputs, targets) in enumerate(train_loader2):
        optimizer_forge.zero_grad()
        outputs = model_forge(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_forge.step()
        forge_losses.append(loss.item())
        diff = abs(loss.item() - eager_losses[step])
        if diff > max_step_diff:
            max_step_diff = diff
        if step in [0, 10, 50, 100, 200, 390]:
            print(f"[FORGE] Step {step}: loss = {loss.item():.4f}, diff vs eager = {diff:.2e}")
            
    test_loss_forge, test_acc_forge = evaluate(model_forge, test_loader2, criterion, torch.device("cpu"))
    print(f"[FORGE Epoch 1] Test Loss: {test_loss_forge:.4f}, Test Acc: {test_acc_forge:.2f}%")
    print(f"Max loss diff across all 391 steps: {max_step_diff:.2e}")

if __name__ == "__main__":
    run_comparison()
