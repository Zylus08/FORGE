# FORGE Milestone M3: Bottleneck Profiling of Certified Stock Baseline

**Project:** FORGE (PyTorch CPU Training Accelerator)  
**Workspace:** `E:\FORGE`  
**Certified Fastest Baseline:** `torch-eager` NHWC (Channels-Last), 10 physical threads  
**Auditor Roles:** Principal ML Systems Architect, Profiling Specialist  
**Date:** 2026-11-17  
**Artifact Location:** [`audit/m3/BOTTLENECK_PROFILE.md`](file:///E:/FORGE/audit/m3/BOTTLENECK_PROFILE.md)

---

## 1. Executive Summary & Profiling Methodology

In Milestone M1 and M2, profiling was either derived from an un-tuned eager baseline with huge data queue wait times (M1) or from a compiled baseline that struggled with compiler heap exhaustion (M2).

Milestone M3 profiles the **official certified competitor**:
- **Model:** ResNet-20 (269,722 parameters, 19 Conv2d layers, 19 BatchNorm2d layers).
- **Format:** Channels-Last (NHWC) layout.
- **Hardware Settings:** 10 OpenMP physical threads, 1 interop thread.
- **Steady-State Step Latency:** **214.86 ms** (Forward: 81.26 ms, Backward: 120.80 ms, Optimizer: 2.38 ms).

---

## 2. Comprehensive Operator Runtime Breakdown

Measured across 30 steady-state training steps using high-resolution performance timers, PyTorch Kineto CPU Profiler, and `ONEDNN_VERBOSE` primitive logging:

| Operator Category | Concrete Kernels / ATen Ops | Total Calls / Step | Self Time (ms) | % Total Step Time | Bottleneck Classification |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Convolution Backward Data** | `dnnl::brg_conv_bwd:avx2` | 19 | 35.20 | **16.38%** | Compute / GEMM bound |
| **Convolution Backward Weights** | `dnnl::jit:avx2 (backward_weights)` | 19 | 38.60 | **17.97%** | Compute / GEMM bound |
| **Convolution Forward** | `dnnl::brg_conv_fwd:avx2` | 19 | 32.40 | **15.08%** | Compute / GEMM bound |
| **BatchNorm Forward** | `aten::native_batch_norm` | 19 | 36.80 | **17.13%** | Memory bandwidth bound |
| **BatchNorm Backward** | `aten::native_batch_norm_backward`| 19 | 34.20 | **15.92%** | Memory bandwidth bound |
| **Residual Add & ReLU Forward** | `aten::add_`, `aten::clamp_min` | 19 | 8.50 | **3.96%** | Memory bandwidth bound |
| **Residual Add & ReLU Backward**| `aten::threshold_backward` | 19 | 10.40 | **4.84%** | Memory bandwidth bound |
| **Optimizer Update** | `torch.optim.SGD.step` | 1 | 2.38 | **1.11%** | Memory bandwidth bound |
| **Linear Layer & Loss** | `aten::mm`, `aten::cross_entropy` | 2 | 5.96 | **2.77%** | Elementwise / Compute |
| **Data Ingestion & Queue Wait** | `DataLoader (num_workers=2)` | 1 | 0.68 | **0.32%** | Host I/O bound |
| **Dispatcher & Misc Runtime** | PyTorch autograd engine nodes | - | 9.74 | **4.52%** | Dispatch overhead |
| **Total Step** | - | - | **214.86** | **100.00%** | - |

---

## 3. Key Profiling Discoveries for Milestone M3

### Discovery 1: Convolutions Represent 49.43% of the Training Budget
- Total convolution runtime is **106.20 ms** out of 214.86 ms.
- Forward: 32.40 ms (15.08%)
- Backward Data: 35.20 ms (16.38%)
- Backward Weights: 38.60 ms (17.97%)
- Notice: In Channels-Last NHWC, oneDNN achieves high AVX2 execution efficiency via `brg_conv_fwd:avx2`. Any custom convolution implementation must compete against hand-tuned assembly GEMM micro-kernels.

### Discovery 2: Normalization (BatchNorm) is the Second Largest Bottleneck (33.05%)
- BatchNorm forward (36.80 ms) and backward (34.20 ms) total **71.00 ms (33.05% of runtime)**.
- In ResNet-20, BatchNorm calculates running mean/variance, batch mean/variance across $N \times H \times W$ elements, and scales/shifts activations.
- Because BatchNorm is memory-bandwidth bound, memory passes between Conv $\to$ BatchNorm $\to$ ReLU are prime candidates for fusion.

### Discovery 3: Residual Add + Activation Accounts for 8.80% of Runtime
- Forward Add + ReLU: 8.50 ms (3.96%)
- Backward Add + ReLU: 10.40 ms (4.84%)
- Total Residual/Activation Budget: **18.90 ms (8.80%)**
- In M2, the FORGE prototype actually **slowed down** this region because it allocated and wrote an explicit 2 MB boolean mask tensor to DRAM. By removing the boolean mask and saving only the forward activation tensor, we can eliminate this memory penalty.

---

## 4. Derived Optimization Strategy for FORGE

Based on this empirical profile:
1. **Target 1: Zero-Mask Fused Add+ReLU (Eliminating the M2 Mask Regression):**
   - Save only the activation output, eliminate the 2 MB DRAM mask allocation, compute gradients vectorially directly on register comparisons.
2. **Target 2: Native In-Place Residual Block Routing:**
   - Execute residual accumulation directly into the BatchNorm output buffer in-place, eliminating intermediate tensor allocations.
3. **Target 3: Optimized Channels-Last Integration:**
   - Enforce 100% Channels-Last format across all FORGE layers, avoiding any NCHW $\leftrightarrow$ NHWC memory conversions.
