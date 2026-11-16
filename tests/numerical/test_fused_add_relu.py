"""Numerical verification of Fused Residual Add + ReLU C++ kernel."""
import pytest
import torch
import torch.nn.functional as F
from forge.backends.forge_backend import (
    get_native_ops,
    fused_add_relu,
    fused_add_relu_cpp,
)


def test_fused_add_relu_parity():
    ops = get_native_ops()
    torch.manual_seed(42)

    for shape in [(128, 16, 32, 32), (128, 32, 16, 16), (128, 64, 8, 8), (64, 64)]:
        x1 = torch.randn(*shape, requires_grad=True)
        r1 = torch.randn(*shape, requires_grad=True)

        x2 = x1.detach().clone().requires_grad_(True)
        r2 = r1.detach().clone().requires_grad_(True)

        # PyTorch reference
        out_ref = F.relu(x1 + r1)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)

        # FORGE fused add + relu (C++ autograd dispatch)
        out_forge = fused_add_relu_cpp(x2, r2)
        out_forge.backward(grad_out)

        # 1. Forward output parity
        fwd_diff = (out_ref - out_forge).abs().max().item()
        assert fwd_diff < 1e-6, f"Forward diff {fwd_diff} exceeds tolerance on {shape}"

        # 2. Backward gradient parity for x
        gx_diff = (x1.grad - x2.grad).abs().max().item()
        assert gx_diff < 1e-6, f"Grad x diff {gx_diff} exceeds tolerance on {shape}"

        # 3. Backward gradient parity for residual
        gr_diff = (r1.grad - r2.grad).abs().max().item()
        assert gr_diff < 1e-6, f"Grad residual diff {gr_diff} exceeds tolerance on {shape}"


def test_fused_add_relu_gradcheck():
    """Analytical gradient check using torch.autograd.gradcheck."""
    from torch.autograd import gradcheck
    x = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
    r = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)

    # Convert to float32 for forge kernel test
    def func(a, b):
        return fused_add_relu(a.float(), b.float()).double()

    assert gradcheck(func, (x, r), eps=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_fused_add_relu_parity()
    test_fused_add_relu_gradcheck()
    print("All fused_add_relu tests passed successfully!")
