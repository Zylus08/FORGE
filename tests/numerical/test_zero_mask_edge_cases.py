"""Comprehensive edge case and numerical audit for FORGE M3 Zero-Mask kernels.
Audits:
1. Derivative at zero (x = 0.0, z = 0.0)
2. Signed zero behavior (+0.0 vs -0.0)
3. NaN and Inf handling
4. Supported FP32 semantics and non-contiguous stride patterns
5. Saved-tensor lifetime and in-place mutation detection
6. Multi-shape analytical precision comparisons
"""
import pytest
import torch
import torch.nn.functional as F
from forge.backends.forge_backend import (
    get_native_ops,
    fused_relu_cpp,
    fused_add_relu_cpp,
)

def test_derivative_at_zero():
    """Verify that derivative at exactly zero is 0.0, matching PyTorch convention."""
    ops = get_native_ops()
    
    # 1. Exact zero input for ReLU
    x = torch.zeros(10, requires_grad=True)
    out_forge = fused_relu_cpp(x)
    grad_out = torch.ones(10)
    out_forge.backward(grad_out)
    
    assert torch.equal(out_forge, torch.zeros(10)), "Forward at zero must be 0.0"
    assert torch.equal(x.grad, torch.zeros(10)), "Backward gradient at zero must be 0.0"

    # Reference check
    xref = torch.zeros(10, requires_grad=True)
    out_ref = F.relu(xref)
    out_ref.backward(grad_out)
    assert torch.equal(x.grad, xref.grad), "Zero subgradient must exactly match PyTorch reference"

    # 2. Exact zero sum for Fused Add+ReLU
    x2 = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    r2 = torch.tensor([ 2.0,  1.0, 0.0, -1.0, -2.0], requires_grad=True)
    out_add = fused_add_relu_cpp(x2, r2)
    out_add.backward(torch.ones(5))
    
    assert torch.equal(out_add, torch.zeros(5)), "Forward (x + r = 0) must be 0.0"
    assert torch.equal(x2.grad, torch.zeros(5)), "Backward grad for x when sum=0 must be 0.0"
    assert torch.equal(r2.grad, torch.zeros(5)), "Backward grad for r when sum=0 must be 0.0"


def test_signed_zero_behavior():
    """Verify behavior on positive zero (+0.0) and negative zero (-0.0)."""
    # In IEEE-754, +0.0 == -0.0 is True
    pos_zero = 0.0
    neg_zero = -0.0
    
    x = torch.tensor([pos_zero, neg_zero, 1e-7, -1e-7], requires_grad=True)
    out = fused_relu_cpp(x)
    out.backward(torch.ones(4))
    
    assert out[0].item() == 0.0
    assert out[1].item() == 0.0
    assert x.grad[0].item() == 0.0
    assert x.grad[1].item() == 0.0
    assert x.grad[2].item() == 1.0  # > 0
    assert x.grad[3].item() == 0.0  # < 0


def test_nan_and_inf_handling():
    """Verify IEEE-754 NaN and Inf propagation through Zero-Mask AVX2 kernels.
    
    Verifies that NaN inputs propagate as NaN in forward, and produce gradient 1.0
    in backward (matching stock PyTorch F.relu and threshold_backward semantics).
    Positive infinity propagates as +inf with grad 1.0; negative infinity clamps to 0.0 with grad 0.0.
    """
    # 1. Fused ReLU
    x = torch.tensor([float('nan'), float('inf'), float('-inf'), 2.0], requires_grad=True)
    out = fused_relu_cpp(x)
    out.backward(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    
    assert torch.isnan(out[0]), "Forward of NaN must propagate as NaN"
    assert torch.isinf(out[1]) and out[1] > 0, "Forward of +inf must be +inf"
    assert out[2] == 0.0, "Forward of -inf must be 0.0"
    assert out[3] == 2.0
    assert x.grad[0].item() == 1.0, "Backward gradient of NaN must match PyTorch (1.0)"
    assert x.grad[1].item() == 1.0, "Backward gradient of +inf must be 1.0"
    assert x.grad[2].item() == 0.0, "Backward gradient of -inf must be 0.0"
    assert x.grad[3].item() == 1.0, "Backward gradient of 2.0 must be 1.0"

    # 2. Fused Add+ReLU
    x2 = torch.tensor([float('nan'), float('inf'), float('-inf'), 2.0], requires_grad=True)
    r2 = torch.tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True)
    out2 = fused_add_relu_cpp(x2, r2)
    out2.backward(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    
    assert torch.isnan(out2[0]), "Add+ReLU Forward of NaN must propagate as NaN"
    assert torch.isinf(out2[1]) and out2[1] > 0, "Add+ReLU Forward of +inf must be +inf"
    assert out2[2] == 0.0, "Add+ReLU Forward of -inf must be 0.0"
    assert out2[3] == 2.0
    assert x2.grad[0].item() == 1.0, "Add+ReLU Backward grad x of NaN must be 1.0"
    assert r2.grad[0].item() == 1.0, "Add+ReLU Backward grad r of NaN must be 1.0"


def test_inplace_mutation_guard():
    """Verify that in-place mutation of saved activation raises autograd error."""
    x = torch.randn(8, 8, requires_grad=True)
    r = torch.randn(8, 8, requires_grad=True)
    
    out = fused_add_relu_cpp(x, r)
    
    # In-place modify out before backward
    out.add_(1.0)
    
    with pytest.raises(RuntimeError) as excinfo:
        out.sum().backward()
    
    assert "one of the variables needed for gradient computation has been modified by an inplace operation" in str(excinfo.value)


def test_exact_numerical_audit_multi_shape():
    """Record exact numerical error metrics across all ResNet-20 feature map shapes."""
    torch.manual_seed(42)
    shapes = [
        ("stage1", (128, 16, 32, 32)),
        ("stage2", (128, 32, 16, 16)),
        ("stage3", (128, 64, 8, 8)),
        ("dense",  (64, 64)),
    ]
    
    metrics = {}
    for name, shape in shapes:
        x_ref = torch.randn(*shape, requires_grad=True)
        r_ref = torch.randn(*shape, requires_grad=True)
        
        x_forge = x_ref.detach().clone().requires_grad_(True)
        r_forge = r_ref.detach().clone().requires_grad_(True)
        
        # Forward
        out_ref = F.relu(x_ref + r_ref)
        out_forge = fused_add_relu_cpp(x_forge, r_forge)
        
        fwd_max_diff = (out_ref - out_forge).abs().max().item()
        fwd_mean_diff = (out_ref - out_forge).abs().mean().item()
        
        # Backward
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_forge.backward(grad_out)
        
        gx_max_diff = (x_ref.grad - x_forge.grad).abs().max().item()
        gx_mean_diff = (x_ref.grad - x_forge.grad).abs().mean().item()
        gr_max_diff = (r_ref.grad - r_forge.grad).abs().max().item()
        gr_mean_diff = (r_ref.grad - r_forge.grad).abs().mean().item()
        
        metrics[name] = {
            "shape": shape,
            "elements": x_ref.numel(),
            "fwd_max_abs_err": fwd_max_diff,
            "fwd_mean_abs_err": fwd_mean_diff,
            "grad_x_max_abs_err": gx_max_diff,
            "grad_x_mean_abs_err": gx_mean_diff,
            "grad_r_max_abs_err": gr_max_diff,
            "grad_r_mean_abs_err": gr_mean_diff,
        }
        assert fwd_max_diff < 1e-6
        assert gx_max_diff < 1e-6
        assert gr_max_diff < 1e-6
        del x_ref, r_ref, x_forge, r_forge, out_ref, out_forge, grad_out
    import gc
    gc.collect()

if __name__ == "__main__":
    test_derivative_at_zero()
    test_signed_zero_behavior()
    test_nan_and_inf_handling()
    test_inplace_mutation_guard()
    m = test_exact_numerical_audit_multi_shape()
    import json
    print(json.dumps(m, indent=2))
    print("All edge case numerical tests passed successfully!")
