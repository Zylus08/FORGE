"""Numerical verification of fused ReLU and fused ReLU+bias kernels against PyTorch reference.

Verifies:
1. Forward output matches F.relu(input) to atol=1e-6
2. Backward gradient matches PyTorch reference to atol=1e-6
3. Multiple shapes: (1,), (128,), (64, 64), (128, 3, 32, 32)
4. Contiguous and non-contiguous inputs
5. Gradient check via torch.autograd.gradcheck
6. Edge cases: all-negative, all-zero, all-positive inputs
7. Mutation safety (input not modified)
8. Multiple sequential calls
9. Different thread counts (torch.set_num_threads(1) and torch.set_num_threads(4))
10. Fused ReLU+Bias forward and backward parity
"""

import os
import pytest
import torch
import torch.nn.functional as F

# Ensure OPENBLAS does not fail on Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from forge.backends.forge_backend import (
    fused_relu,
    fused_relu_bias,
    FusedReLUFunction,
    FusedReLUBiasFunction,
    FusedReLU,
    FusedReLUBias,
    forge_replace_ops,
    ForgePatchContext,
)


@pytest.fixture(autouse=True)
def setup_seed():
    torch.manual_seed(42)


def test_forward_parity_shapes():
    shapes = [
        (1,),
        (128,),
        (64, 64),
        (128, 3, 32, 32),
        (16, 64, 8, 8),
    ]
    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float32)
        expected = F.relu(x)
        actual = fused_relu(x)
        assert torch.allclose(actual, expected, atol=1e-6), f"Forward mismatch on shape {shape}"


def test_backward_parity_shapes():
    shapes = [
        (1,),
        (128,),
        (64, 64),
        (128, 3, 32, 32),
    ]
    for shape in shapes:
        x_ref = torch.randn(shape, dtype=torch.float32, requires_grad=True)
        x_act = x_ref.detach().clone().requires_grad_(True)

        y_ref = F.relu(x_ref)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)

        y_act = fused_relu(x_act)
        y_act.backward(grad_out)

        assert torch.allclose(x_act.grad, x_ref.grad, atol=1e-6), f"Backward mismatch on shape {shape}"


def test_non_contiguous_inputs():
    # Create non-contiguous tensor by slicing or transposing
    x_base = torch.randn(32, 64, dtype=torch.float32)
    x = x_base.t()  # non-contiguous (64, 32)
    assert not x.is_contiguous()

    expected = F.relu(x)
    actual = fused_relu(x)
    assert torch.allclose(actual, expected, atol=1e-6)

    # Test backward on non-contiguous
    x_ref = x.clone().requires_grad_(True)
    x_act = x.clone().requires_grad_(True)
    grad_out = torch.randn_like(x)

    F.relu(x_ref).backward(grad_out)
    fused_relu(x_act).backward(grad_out)
    assert torch.allclose(x_act.grad, x_ref.grad, atol=1e-6)


def test_edge_cases():
    shapes = [(32, 32), (8, 16, 4, 4)]
    for shape in shapes:
        # All negative
        x_neg = -torch.rand(shape) - 0.1
        assert torch.all(fused_relu(x_neg) == 0.0)

        # All zero
        x_zero = torch.zeros(shape)
        assert torch.all(fused_relu(x_zero) == 0.0)

        # All positive
        x_pos = torch.rand(shape) + 0.1
        assert torch.allclose(fused_relu(x_pos), x_pos, atol=1e-6)


def test_mutation_safety():
    x = torch.randn(16, 16, dtype=torch.float32)
    x_copy = x.clone()
    _ = fused_relu(x)
    assert torch.equal(x, x_copy), "Input tensor was mutated in-place by fused_relu"


def test_sequential_calls():
    x = torch.randn(32, 32, dtype=torch.float32, requires_grad=True)
    y1 = fused_relu(x)
    y2 = fused_relu(y1)
    y3 = fused_relu(y2)
    loss = y3.sum()
    loss.backward()

    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = F.relu(F.relu(F.relu(x_ref)))
    y_ref.sum().backward()

    assert torch.allclose(y3, y_ref, atol=1e-6)
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)


def test_different_thread_counts():
    for threads in [1, 2, 4]:
        torch.set_num_threads(threads)
        x = torch.randn(64, 64, dtype=torch.float32, requires_grad=True)
        y = fused_relu(x)
        grad = torch.randn_like(y)
        y.backward(grad)

        x_ref = x.detach().clone().requires_grad_(True)
        F.relu(x_ref).backward(grad)
        assert torch.allclose(x.grad, x_ref.grad, atol=1e-6)


def test_gradcheck():
    # gradcheck requires double precision for finite-difference accuracy
    # We test on float64 or small float32 perturbations
    # Custom function takes float32, let's verify numerical gradient manually or using float64 if supported
    x = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
    # Avoid inputs exactly at 0 where ReLU is non-differentiable
    x.data[x.data.abs() < 0.05] += 0.1

    def func(inp):
        return FusedReLUFunction.apply(inp)

    # PyTorch gradcheck
    # eps=1e-3, atol=1e-2 for float32 finite differences
    assert torch.autograd.gradcheck(func, (x,), eps=1e-3, atol=1e-2, nondet_tol=1e-5)


def test_fused_relu_bias_parity():
    # N, C, H, W
    x = torch.randn(8, 16, 7, 7, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(16, dtype=torch.float32, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    b_ref = bias.detach().clone().requires_grad_(True)

    # Reference computation: x + bias.view(1, C, 1, 1) then ReLU
    y_ref = F.relu(x_ref + b_ref.view(1, 16, 1, 1))
    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)

    # Fused computation
    y_act = fused_relu_bias(x, bias)
    y_act.backward(grad_out)

    assert torch.allclose(y_act, y_ref, atol=1e-6), "fused_relu_bias forward mismatch"
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6), "fused_relu_bias grad_x mismatch"
    assert torch.allclose(bias.grad, b_ref.grad, atol=1e-5), "fused_relu_bias grad_bias mismatch"


def test_forge_replace_ops():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),
    )
    forge_replace_ops(model)
    assert isinstance(model[1], FusedReLU)
    assert isinstance(model[3], FusedReLU)

    inp = torch.randn(4, 3, 8, 8)
    out = model(inp)
    assert out.shape == (4, 16, 8, 8)


def test_forge_patch_context():
    with ForgePatchContext():
        x = torch.randn(10, 10, requires_grad=True)
        y = F.relu(x)
        assert y is not None
        y.sum().backward()
        assert x.grad is not None
    # Verify F.relu restored
    assert F.relu != fused_relu
