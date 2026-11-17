"""FORGE Native Backend: High-Performance C++20 Autograd & Fused Residual Kernels."""

import os
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

_native_ops = None

def get_native_ops():
    """Load or return the compiled native C++ extension."""
    global _native_ops
    if _native_ops is not None:
        return _native_ops

    # 1. Try importing pre-built extension
    try:
        from forge import _forge_native
        _native_ops = _forge_native
        return _native_ops
    except ImportError:
        pass

    try:
        import _forge_native
        _native_ops = _forge_native
        return _native_ops
    except ImportError:
        pass

    # 2. Try JIT compiling via torch.utils.cpp_extension.load
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent.parent.parent
    src = root / "native" / "src" / "fused_ops_m2.cpp"
    inc = root / "native" / "include"

    os.environ.setdefault("TEMP", r"E:\FORGE\.tmp")
    os.environ.setdefault("TMP", r"E:\FORGE\.tmp")
    os.makedirs(r"E:\FORGE\.tmp", exist_ok=True)

    _native_ops = load(
        name="forge_native_jit",
        sources=[str(src)],
        extra_include_paths=[str(inc)],
        extra_cflags=["/std:c++20", "/O2", "/arch:AVX2", "/EHsc"],
        verbose=False,
    )
    return _native_ops


# ---------------------------------------------------------------------------
# C++ Native Autograd Dispatch (Zero Python Function Boundary Overhead)
# ---------------------------------------------------------------------------

def fused_relu_cpp(x: torch.Tensor) -> torch.Tensor:
    """High-speed native C++ autograd-dispatched fused ReLU (Zero-Mask M3)."""
    ops = get_native_ops()
    if hasattr(ops, "native_autograd_fused_relu_v3"):
        return ops.native_autograd_fused_relu_v3(x)
    if hasattr(ops, "native_autograd_fused_relu"):
        return ops.native_autograd_fused_relu(x)
    return FusedReLUFunction.apply(x)


def fused_add_relu_cpp(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """High-speed native C++ autograd-dispatched fused residual add + ReLU (Zero-Mask M3)."""
    ops = get_native_ops()
    if hasattr(ops, "native_autograd_fused_add_relu_v3"):
        return ops.native_autograd_fused_add_relu_v3(x, residual)
    if hasattr(ops, "native_autograd_fused_add_relu"):
        return ops.native_autograd_fused_add_relu(x, residual)
    return FusedAddReLUFunction.apply(x, residual)


# ---------------------------------------------------------------------------
# Legacy Python Autograd Functions (for fallback and ablation comparison)
# ---------------------------------------------------------------------------

class FusedReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ops = get_native_ops()
        output, mask = ops.fused_relu_forward(input_tensor)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (mask,) = ctx.saved_tensors
        ops = get_native_ops()
        grad_input = ops.fused_relu_backward(grad_output, mask)
        return grad_input


class FusedAddReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        ops = get_native_ops()
        output, mask = ops.fused_add_relu_forward(x, residual)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask,) = ctx.saved_tensors
        ops = get_native_ops()
        grads = ops.fused_add_relu_backward(grad_output, mask)
        return grads[0], grads[1]


class FusedReLUBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        ops = get_native_ops()
        output, mask = ops.fused_relu_bias_forward(input_tensor, bias)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask,) = ctx.saved_tensors
        ops = get_native_ops()
        grad_input, grad_bias = ops.fused_relu_bias_backward(grad_output, mask)
        return grad_input, grad_bias


def fused_relu(x: torch.Tensor) -> torch.Tensor:
    return fused_relu_cpp(x)


def fused_add_relu(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return fused_add_relu_cpp(x, residual)


def fused_relu_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return FusedReLUBiasFunction.apply(x, bias)


class FusedReLUBias(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_relu_bias(x, self.bias)


class FusedReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_relu_cpp(x)


class ForgePatchContext:
    """Context manager patching torch.nn.functional.relu with native fused_relu."""

    def __init__(self):
        self._orig_relu = None

    def __enter__(self):
        self._orig_relu = F.relu
        F.relu = fused_relu_cpp
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig_relu is not None:
            F.relu = self._orig_relu


def forge_replace_ops(model: nn.Module) -> nn.Module:
    """Recursively replaces nn.ReLU with FusedReLU in model submodules."""
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, FusedReLU(inplace=child.inplace))
        else:
            forge_replace_ops(child)
    return model
