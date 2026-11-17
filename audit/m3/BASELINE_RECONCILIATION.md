# FORGE Milestone M3: Baseline Reconciliation & Performance Forensics

**Project:** FORGE (PyTorch CPU Training Accelerator)  
**Workspace:** `E:\FORGE`  
**Starting Commit:** [`c4cfc57`](file:///E:/FORGE)  
**Previous Implementation Commit:** [`917049a`](file:///E:/FORGE)  
**Auditor Roles:** Principal CPU Performance Architect, PyTorch Internals Engineer, Independent Benchmark Auditor  
**Date:** 2026-11-17  
**Artifact Location:** [`audit/m3/BASELINE_RECONCILIATION.md`](file:///E:/FORGE/audit/m3/BASELINE_RECONCILIATION.md)

---

## 1. Executive Summary & Forensic Findings

Milestone M3 began with a critical forensic audit of the performance claims and benchmark results delivered at the end of Milestone M2. 

### The Core Contradiction in M2 Reporting:
In Milestone M2, early 1-epoch summaries claimed a **2.80× end-to-end wall-clock speedup** over stock eager PyTorch (111.78s vs. 313.36s). However, when full 5-epoch training was completed across all four backends under identical test conditions ([`audit/m2/BENCHMARK_RESULTS.csv`](file:///E:/FORGE/audit/m2/BENCHMARK_RESULTS.csv)), the total 5-epoch elapsed times revealed a stark contradiction:

| Backend Configuration | Total 5-Epoch Time (s) | Mean Epoch Time (s) | Steady-State Median Step (ms) | Speedup vs Fastest Stock |
| :--- | :---: | :---: | :---: | :---: |
| **`torch-tuned` (B1 Stock)** | **488.01 s** | **97.60 s** | **218.6 ms** | **1.00× (Fastest Stock Baseline)** |
| **`torch-eager` (B0 Stock)** | 521.90 s | 104.38 s | 273.6 ms | 0.94× |
| **`forge` (M2 Native)** | **577.03 s** | **115.41 s** | **290.1 ms** | **0.85× (Slower than Stock Tuned)** |
| **`torch-compiled` (B2 Stock)** | 625.39 s | 125.08 s | 321.7 ms | 0.78× |

### Forensic Reality:
1. **FORGE M2 is SLOWER than the fastest stock baseline:** FORGE M2 (577.03 s) is **1.18× slower (0.85× throughput)** than `torch-tuned` (488.01 s).
2. **The "2.80× speedup" claim was measured against an un-tuned, handicapped reference:** In early M2 tests, B0 eager took 313.36 s on epoch 1 because it ran in un-transposed NCHW with unconstrained OS thread pools suffering from P-core/E-core scheduling contention (>700 ms/step). When B0 runs with standard warm DataLoader workers, its step time drops to ~244–273 ms (109.98 s/epoch).
3. **The official stock baseline hurdle is B1 (`torch-tuned`), NOT B2 (`torch-compiled`):** TorchInductor on CPU training is slower than tuned eager execution (625.39 s vs. 488.01 s) due to oneDNN fallback boundaries and OpenMP parallel region dispatch latency. Comparing FORGE against B2 was comparing against a slower competitor.

---

## 2. Root Cause Analysis of the FORGE Performance Penalty

Why did the FORGE M2 native prototype take 577.03 s, losing to `torch-tuned` (488.01 s)? Micro-architectural profiling and source inspection uncovered three distinct root causes:

### Root Cause 1: Memory Layout Asymmetry in Benchmark Harness
In `forge/benchmarking/run_baselines.py`:
- In `configure_backend()`:
  ```python
  elif backend == "torch-tuned":
      torch.set_num_threads(physical_cores)
      torch.set_num_interop_threads(1)
      model = model.to(memory_format=torch.channels_last) # <-- Converted to NHWC!
  elif backend == "forge":
      torch.set_num_threads(physical_cores)
      torch.set_num_interop_threads(1)
      model = apply_forge_backend(model) # <-- Remained in NCHW!
  ```
- In `timed_train_epoch()`:
  ```python
  if backend == "torch-tuned":
      inputs = inputs.to(memory_format=torch.channels_last) # <-- Converted to NHWC!
  ```
`torch-tuned` ran entirely in **Channels-Last (NHWC)** format, unlocking oneDNN's AVX2 Block-Row-Group GEMM convolution kernels (`brg_conv_fwd:avx2`). Conversely, `forge` was evaluated in **Contiguous (NCHW)** format, forcing oneDNN to execute expensive internal memory reorders (`abcd` $\leftrightarrow$ `aBcd8b`) on every single convolution forward and backward pass!

### Root Cause 2: The Boolean Activation Mask Penalty
In `native/src/fused_ops_m2.cpp`:
The M1/M2 prototype attempted to accelerate ReLU by generating and saving a packed 8-bit boolean mask array during the forward pass:
```cpp
// Forward pass: 128x16x32x32 floats -> writes 2 MB float out + 2 MB bool mask to DRAM!
__m256 cmp = _mm256_cmp_ps(v, zero8, _CMP_GT_OQ);
__m128i p16 = _mm_packs_epi32(lo, hi);
__m128i p8  = _mm_packs_epi16(p16, p16);
_mm_storel_epi64(reinterpret_cast<__m128i*>(m_ptr + i), b8);
```
**Empirical Microbenchmark Comparison (`batch_size=128, channels=16, 32x32`):**
- **PyTorch Native `F.relu`:** Forward = **0.180 ms**, Backward = **1.662 ms**, Total = **1.842 ms**
- **FORGE `fused_relu_cpp`:** Forward = **0.889 ms**, Backward = **2.135 ms**, Total = **3.025 ms**
- **Result:** FORGE's custom ReLU was **1.64× SLOWER** than stock PyTorch!
  - Forward was **4.9× slower** due to writing 2 MB of mask data to DRAM.
  - Backward was slower due to loading the mask from DRAM and running vector sign-extend instructions (`_mm256_cvtepi8_epi32`).
- **Why Stock PyTorch Wins:** PyTorch's native `threshold_backward(grad, output, 0.0)` does NOT allocate a mask tensor. It checks `output > 0` directly in CPU registers during backward, avoiding DRAM round-trips entirely.

### Root Cause 3: `fused_add_relu` Allocation Overhead
- **PyTorch Native `F.relu(x + res)`:** Forward = **0.890 ms**, Backward = **2.676 ms**, Total = **3.567 ms**
- **FORGE `fused_add_relu_cpp`:** Forward = **1.417 ms**, Backward = **3.678 ms**, Total = **5.095 ms**
- **Result:** FORGE was **1.43× SLOWER** than stock PyTorch!
  - It allocated two output tensors in forward (`out` and `mask`) and returned two gradient tensors in backward, incurring continuous dynamic heap allocator overhead.

---

## 3. Environment & Hardware Health Verification

- **Host Processor:** Intel Core i7-12650H (Alder Lake-H: 6 Performance cores with HyperThreading + 4 Efficient cores = 10 physical cores, 16 logical threads).
- **Instruction Extensions:** AVX2, FMA, SSE4.2 (confirmed; AVX-512 is not present on Alder Lake client chips).
- **RAM:** 16.0 GB DDR4-3200 (Dual Channel).
- **Storage Profile:**
  - `C:\`: ~61 MB free (**CRITICALLY LOW**).
  - `E:\`: ~111 GB free (**PRIMARY WORKSPACE**).
  - **Constraint Enforced:** All temp files, build directories, torchvision datasets, and caches MUST reside on `E:\FORGE\.tmp`.
- **Software Stack:**
  - Python: 3.14.2 (conda-forge, MSC v.1944 64-bit).
  - PyTorch: 2.15.0+cpu (MKL enabled, oneDNN v3.13.3 enabled, OpenMP enabled).
  - C++ Toolchain: MSVC v14.43.34808 (`cl.exe`), Windows Kits 10 SDK (10.0.22621.0).
- **Existing Test Health:** 24/24 tests passed in 5.13s ([`audit/m2/test_results.txt`](file:///E:/FORGE/audit/m2/test_results.txt)).

---

## 4. Canonical Workload Contract & Baseline Freeze

The scientific workload contract is strictly frozen:
- **Model:** ResNet-20 (`workloads/models/resnet20.py`, 269,722 trainable parameters).
- **Dataset:** CIFAR-10 (50,000 train, 10,000 test, 32×32×3 RGB).
- **Batch Size:** 128 (391 steps per epoch).
- **Optimizer:** SGD(lr=0.01, momentum=0.9, weight_decay=1e-4).
- **Loss:** CrossEntropyLoss.
- **Precision:** FP32 (no mixed precision, no quantization shortcuts).
- **Target Accuracy:** $\ge 85.0\%$ top-1 validation accuracy at 160-epoch horizon.

---

## 5. Milestone M3 Scientific Ground Rules

1. **Official Hurdle:** The official stock baseline is **`torch-tuned` (B1)** with 10 OpenMP threads and Channels-Last NHWC layout (488.01s across 5 epochs, 218.6 ms steady-state step time).
2. **Speedup Ground Truth:** Any claimed speedup must be measured against **`torch-tuned` (B1)**, not against un-tuned eager (B0) or compiled (B2).
3. **No Fabricated Speedups:** If FORGE achieves a speedup, it must be supported by reproducible end-to-end measurements. If the 2.5× objective is not mathematically or practically achieved, we will state **NOT ACHIEVED** with full scientific attribution.
4. **Independent Engineering Verdict:** The engineering milestone gate (PASS/PARTIAL/BLOCKED) will be evaluated separately from the 2.5× target (ACHIEVED/NOT ACHIEVED).
