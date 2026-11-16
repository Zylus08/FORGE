# FORGE Numerical Accuracy Contract

## Motivation

FORGE backends replace or fuse PyTorch operators. Any replacement must produce results that are **numerically equivalent** to the `torch-eager` baseline within defined tolerances. This document is the binding contract.

## Tolerance Definitions

| Quantity | Absolute tolerance (`atol`) | Relative tolerance (`rtol`) | Notes |
|----------|-----------------------------|-----------------------------|-------|
| Forward activations (fp32) | 1e-5 | 1e-4 | Per-element |
| Backward gradients (fp32) | 1e-4 | 1e-3 | Per-element |
| Loss scalar (fp32) | 1e-5 | 1e-4 | Scalar comparison |
| Weight updates (fp32) | 1e-4 | 1e-3 | After one optimizer step |
| Accumulation over N steps | N × 1e-5 | 1e-3 | Drift budget per step |

All comparisons use `torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)`.

## What Must Match

For every backend, the following must pass the tolerance check against `torch-eager` on the **canonical verification workload** (single-batch, seed 42):

1. **Forward pass output** — final layer activations.
2. **Loss value** — scalar loss from the criterion.
3. **Gradient of every leaf parameter** — `param.grad` after `loss.backward()`.
4. **Parameter values after one optimizer step** — using SGD with `lr=0.01, momentum=0`.

## What Is Permitted to Differ

- **Operation order** (associativity changes are acceptable if within tolerance).
- **Intermediate activations** that are not user-visible (e.g., fused kernel internal temps).
- **Timing and memory layout** — backends may choose different strides/contiguity.
- **Non-determinism across runs** — reproducibility is enforced only via fixed seeds within a single process.

## Verification Procedure

```bash
# Run the accuracy suite for a specific backend
pytest tests/numerical/ -k "forge-onednn" -v

# Check a custom backend
python -m forge.verify --backend my-backend --workload resnet18 --seed 42
```

The verifier:
1. Runs `torch-eager` with seed 42, captures reference tensors.
2. Runs the candidate backend with the same seed and inputs.
3. Calls `torch.testing.assert_close` on every quantity in "What Must Match".
4. Reports pass/fail per quantity and prints max absolute and relative errors.

## Failure Policy

A backend that **fails** the accuracy contract:
- Must **not** be published in a benchmark comparison.
- Must be labeled `EXPERIMENTAL` in the registry.
- Requires a root-cause issue filed before merging.

## Mixed Precision

fp16 and bf16 tolerances are TBD (outside M1 scope). Do not run accuracy checks in mixed precision at M1 without first updating this document.
