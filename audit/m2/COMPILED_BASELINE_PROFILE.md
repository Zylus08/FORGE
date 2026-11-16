# FORGE Milestone M2: Compiled Baseline Profiling & Bottleneck Analysis

**Date:** 2026-11-17  
**Auditor:** Principal ML Systems Architect & Profiler Specialist  
**Artifact Location:** [`audit/m2/COMPILED_BASELINE_PROFILE.md`](file:///E:/FORGE/audit/m2/COMPILED_BASELINE_PROFILE.md)  
**Trace Artifact:** [`benchmarks/results/profile_compiled.json`](file:///E:/FORGE/benchmarks/results/profile_compiled.json)

---

## 1. Profiling Methodology

In Milestone M1, bottleneck percentages were derived solely from eager execution. In Milestone M2, we profiled the official compiled baseline **B2 (`torch.compile(mode="default", backend="inductor")`)** over 10 training steps using the PyTorch Kineto CPU Profiler on ResNet-20 ($N=128, C=3, H=32, W=32$, 10 OpenMP threads).

---

## 2. Top Operator Breakdown (Compiled Baseline B2)

Total Profiled Self CPU Time: **4,927.86 ms** (10 steps)

| Rank | Operator / Runtime Event | Calls | Self CPU (s) | Self CPU % | CPU Total (s) | CPU Total % | Classification |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `aten::convolution_backward` | 190 | 2.183 | **44.29%** | 2.214 | 44.92% | Compute / GEMM bound |
| **2** | `aten::mkldnn_convolution` | 190 | 1.056 | **21.43%** | 1.066 | 21.64% | Compute / oneDNN bound |
| **3** | `CompiledFxGraph (Backward Call)` | 10 | 0.817 | **16.58%** | 3.032 | 61.52% | Fused Inductor C++ Loops |
| **4** | `CompiledFxGraph (Forward Call)` | 10 | 0.518 | **10.52%** | 1.598 | 32.42% | Fused Inductor C++ Loops |
| **5** | `DataLoader Iter Next` | 10 | 0.093 | **1.88%** | 0.093 | 1.89% | Host I/O / Prefetch |
| **6** | `Optimizer.step#SGD.step` | 10 | 0.040 | **0.81%** | 0.098 | 1.98% | Memory bandwidth bound |
| **7** | `aten::copy_` | 800 | 0.028 | **0.56%** | 0.028 | 0.56% | Memory movement |
| **8** | `Optimizer.zero_grad` | 10 | 0.017 | **0.34%** | 0.017 | 0.34% | Memory initialization |
| **9** | `aten::resize_` | 560 | 0.015 | **0.31%** | 0.015 | 0.31% | Memory management |
| **10** | `aten::add_ / aten::add` | 1770 | 0.029 | **0.59%** | 0.029 | 0.59% | Elementwise arithmetic |
| - | **Other Operations** | - | 0.132 | **2.69%** | - | - | Miscellaneous |

---

## 3. High-Level Architectural Findings

### 1. Convolution Operations Form the Unfused Core (65.72% of Runtime)
TorchInductor **does not fuse convolutions** on CPU during training. Instead, it generates callouts to oneDNN (`aten::mkldnn_convolution`) for forward and PyTorch's native convolution backward (`aten::convolution_backward`) for backward.
- `aten::convolution_backward`: 44.29%
- `aten::mkldnn_convolution`: 21.43%
- Total Convolution Cost: **65.72%**

### 2. TorchInductor Fuses Normalization and Elementwise Activations (27.10% of Runtime)
Unlike eager mode where `aten::native_batch_norm` and `aten::clamp_min` appear as distinct operators, TorchInductor bundles BatchNorm, ReLU, and residual additions into generated C++ OpenMP kernels (`CompiledFxGraph` forward: 10.52%, backward: 16.58%).

### 3. Data Ingestion Overhead is Negligible (1.88%)
With `num_workers=2` and `persistent_workers=True`, data loading accounts for only 1.88% (93 ms across 10 steps = ~9.3 ms/step), confirming that data loading is not the bottleneck on this system.

---

## 4. Eager vs. Compiled Comparison

| Functional Category | Eager Baseline (B0) % | Compiled Baseline (B2) % | Inductor Transformation Mechanism |
| :--- | :---: | :---: | :--- |
| **Convolution Forward** | 18.40% | **21.43%** | Delegated to oneDNN direct kernel |
| **Convolution Backward** | 39.81% | **44.29%** | Delegated to ATen convolution backward |
| **BatchNorm Forward** | 3.97% | *Fused into graph* | Combined with activation loops |
| **BatchNorm Backward** | 7.73% | *Fused into graph* | Combined with gradient reduction loops |
| **ReLU Activations** | 4.93% | *Fused into graph* | Integrated into preceding elementwise pass |
| **Residual Addition** | 2.65% | *Fused into graph* | In-place accumulator in C++ loop |
| **Generated C++ Graph Loops** | 0.00% | **27.10%** | Replaces individual elementwise dispatches |
| **Optimizer Step** | 1.71% | **0.81%** | Vectorized loop execution |
| **Data Ingestion** | 15.98% (with w=0) | **1.88%** (with w=2) | Multi-process worker prefetching |

---

## 5. Implications for FORGE Native Acceleration

1. **Activation Kernels in Isolation Cannot Move the Needle:** Because TorchInductor already fuses elementwise activations into loop bodies, an isolated custom ReLU kernel cannot yield a speedup against compiled PyTorch.
2. **Residual Path Fusion is High Value:** In eager execution, residual addition + ReLU represents ~7.5% of total runtime and allocates redundant intermediate buffers. Fusing residual add + activation (`fused_add_relu`) eliminates buffer allocations and unifies gradient backward propagation.
3. **C++ Autograd Dispatch is Essential:** In M1, the Python `torch.autograd.Function` wrapper incurred ~0.3 ms fixed overhead per call (7,429 calls per epoch = 2.2 seconds of pure Python dispatch latency). Registering custom C++ Autograd functions directly in PyTorch eliminates this barrier entirely.
