"""Integration smoke tests for training workloads across backends."""
import pytest
import math
import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.integration.forge_trainer import ForgeTrainer


@pytest.mark.parametrize("backend", ["torch-eager", "torch-tuned", "forge"])
def test_training_smoke_2_steps(backend):
    """Runs 2 steps of training on backend, checks loss decreases or is finite, and accuracy is computable."""
    torch.manual_seed(42)

    # 1. Setup model, criterion, optimizer
    raw_model = resnet20()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.01, momentum=0.9)

    # Apply backend configurations
    if backend == "torch-tuned":
        torch.set_num_threads(torch.get_num_threads())
        torch.backends.mkl.enabled = True
        torch.backends.mkldnn.enabled = True
        model = raw_model
    elif backend == "forge":
        trainer = ForgeTrainer(raw_model, criterion, optimizer)
        model = trainer.model
        optimizer = trainer.optimizer
    else:  # torch-eager
        model = raw_model

    model.train()

    # Synthetic batch of 2 steps: batch_size=4, 3 channels, 32x32
    inputs = [torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)]
    targets = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])]

    losses = []
    accuracies = []

    for step in range(2):
        optimizer.zero_grad()
        output = model(inputs[step])
        loss = criterion(output, targets[step])

        # Check loss is finite
        loss_val = loss.item()
        assert not math.isnan(loss_val) and not math.isinf(loss_val), f"Loss is not finite: {loss_val}"
        losses.append(loss_val)

        # Backward
        loss.backward()
        optimizer.step()

        # Compute accuracy
        preds = output.argmax(dim=1)
        correct = (preds == targets[step]).sum().item()
        acc = correct / targets[step].size(0)
        assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"
        accuracies.append(acc)

    # Verify 2 steps were recorded and valid
    assert len(losses) == 2
    assert len(accuracies) == 2
    assert all(math.isfinite(l) for l in losses)
