"""Deterministic paired training test: PyTorch eager vs FORGE backend.
Verifies exact mathematical equivalence across:
- Forward activations
- Loss values
- Backward gradients
- Optimizer momentum buffers
- Updated model parameters
- BatchNorm running statistics
"""
import pytest
import torch
import torch.nn as nn
from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds
from forge.integration.forge_trainer import apply_forge_backend


def test_paired_training_exact_parity():
    """Verify 10 steps of paired training produce identical outputs, gradients, and weights."""
    set_seeds(42)
    model_ref = resnet20()

    set_seeds(42)
    model_forge = resnet20()

    # Verify initial weights are strictly identical
    for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_forge.named_parameters()):
        assert torch.equal(p1, p2), f"Initial parameter mismatch in {n1}"

    model_forge = apply_forge_backend(model_forge)

    criterion = nn.CrossEntropyLoss()
    opt_ref = torch.optim.SGD(model_ref.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    opt_forge = torch.optim.SGD(model_forge.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # Run 10 training steps with identical deterministic pseudo-batches
    torch.manual_seed(100)
    for step in range(10):
        model_ref.train()
        model_forge.train()

        x = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 10, (32,))

        opt_ref.zero_grad()
        opt_forge.zero_grad()

        out_ref = model_ref(x)
        out_forge = model_forge(x)

        # 1. Forward activations parity
        fwd_diff = (out_ref - out_forge).abs().max().item()
        assert fwd_diff < 1e-5, f"Step {step}: Forward activation mismatch: {fwd_diff}"

        loss_ref = criterion(out_ref, y)
        loss_forge = criterion(out_forge, y)

        # 2. Scalar loss parity
        loss_diff = abs(loss_ref.item() - loss_forge.item())
        assert loss_diff < 1e-5, f"Step {step}: Loss mismatch: {loss_diff}"

        loss_ref.backward()
        loss_forge.backward()

        # 3. Parameter gradients parity
        for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_forge.named_parameters()):
            assert p1.grad is not None and p2.grad is not None, f"Missing gradient in {n1}"
            gdiff = (p1.grad - p2.grad).abs().max().item()
            assert gdiff < 1e-5, f"Step {step}: Gradient mismatch in {n1}: {gdiff}"

        opt_ref.step()
        opt_forge.step()

        # 4. Parameters after optimizer step parity
        for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_forge.named_parameters()):
            pdiff = (p1 - p2).abs().max().item()
            assert pdiff < 1e-5, f"Step {step}: Parameter mismatch after step in {n1}: {pdiff}"

        # 5. BatchNorm running statistics parity
        for (n1, b1), (n2, b2) in zip(model_ref.named_buffers(), model_forge.named_buffers()):
            bdiff = (b1 - b2).abs().max().item()
            assert bdiff < 1e-5, f"Step {step}: Buffer mismatch in {n1}: {bdiff}"

        # 6. Optimizer momentum buffer parity
        for p1, p2 in zip(model_ref.parameters(), model_forge.parameters()):
            state1 = opt_ref.state[p1]
            state2 = opt_forge.state[p2]
            if "momentum_buffer" in state1:
                mb1 = state1["momentum_buffer"]
                mb2 = state2["momentum_buffer"]
                mb_diff = (mb1 - mb2).abs().max().item()
                assert mb_diff < 1e-5, f"Step {step}: Momentum buffer mismatch: {mb_diff}"


if __name__ == "__main__":
    test_paired_training_exact_parity()
    print("Paired training test passed successfully!")
