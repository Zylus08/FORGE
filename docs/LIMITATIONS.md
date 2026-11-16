# FORGE M1 — Known Limitations

This document lists limitations that are known at Milestone M1 and are explicitly **out of scope** for M1 deliverables. Each item notes the planned milestone for resolution.

## 1. No GPU Support

FORGE targets CPU-only training. All backends assume `device="cpu"`. Attempting to run on a CUDA device will raise `NotImplementedError`. *(Planned: post-M1 roadmap item)*

## 2. Native Kernels Are Stubs

`native/src/fused_relu_linear_backward.cpp` is a placeholder at M1. The CMakeLists.txt builds it, but it contains only the function signature; actual fusion is implemented at M2. *(Planned: M2)*

## 3. Single-Node Only

FORGE does not support distributed training (`torch.distributed`, DDP, FSDP). All benchmarks are single-process, single-node. *(Planned: post-M2)*

## 4. Limited Workload Coverage

M1 baselines cover only:
- Synthetic linear stack (fully connected, variable depth)
- ResNet-18 (image classification, synthetic data)

RNNs, Transformers, and graph networks are out of scope at M1. *(Planned: M2–M3)*

## 5. No Mixed-Precision Benchmarks

All benchmarks run in fp32. fp16 and bf16 paths are not tested or validated. The accuracy contract does not cover them. *(Planned: M2)*

## 6. Windows-Only CI at M1

The hardware audit and primary benchmarks are run on Windows. Linux CI is not configured at M1. *(Planned: M2 adds Linux runner)*

## 7. No Autotuning

Backend hyperparameters (tile sizes, thread counts for native kernels) are fixed. No autotuning or hardware-adaptive selection is performed at M1. *(Planned: M3)*

## 8. Benchmark Results Are Not Peer-Reviewed

M1 results are internal engineering measurements, not publication-grade numbers. They have not been reviewed for statistical rigor beyond the protocol in `docs/BENCHMARK_PROTOCOL.md`.

## 9. `torch.compile` Backend Is Informational Only

The `torch-compile` backend is included for reference but is not part of the M1 accuracy contract. Its output may differ from `torch-eager` within PyTorch's own tolerances.

## 10. Power-Plan Sensitivity Not Mitigated

Benchmark results depend on the active Windows power plan ("Balanced" vs. "High Performance"). The hardware audit records the active plan but does not enforce it. Users must set "High Performance" manually before running benchmarks for comparable results.
