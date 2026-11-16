"""FORGE Backend Registry and Factory."""

from typing import Any, Callable, Dict

from .forge_backend import (
    FusedReLU,
    FusedReLUBias,
    FusedReLUFunction,
    FusedReLUBiasFunction,
    ForgePatchContext,
    forge_replace_ops,
    fused_relu,
    fused_relu_bias,
    get_native_ops,
)

def get_backend(name: str):
    """Factory function returning backend configuration and setup hooks."""
    name_clean = name.lower().strip()
    if name_clean in ("forge", "forge-native"):
        return {
            "name": "forge",
            "type": "native_accelerated",
            "replace_ops": forge_replace_ops,
            "patch_context": ForgePatchContext,
            "fused_relu": fused_relu,
        }
    elif name_clean in ("torch-eager", "eager"):
        return {
            "name": "torch-eager",
            "type": "stock_eager",
        }
    elif name_clean in ("torch-tuned", "tuned"):
        return {
            "name": "torch-tuned",
            "type": "stock_tuned",
        }
    elif name_clean in ("torch-compiled", "compiled", "inductor"):
        return {
            "name": "torch-compiled",
            "type": "stock_compiled",
        }
    else:
        raise ValueError(f"Unknown backend '{name}'. Supported: torch-eager, torch-tuned, torch-compiled, forge")

__all__ = [
    "get_backend",
    "FusedReLU",
    "FusedReLUBias",
    "FusedReLUFunction",
    "FusedReLUBiasFunction",
    "ForgePatchContext",
    "forge_replace_ops",
    "fused_relu",
    "fused_relu_bias",
    "get_native_ops",
]
