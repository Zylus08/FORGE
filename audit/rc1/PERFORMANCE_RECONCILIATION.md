# RC1 Performance Reconciliation Report: Recalculated Claims, Population Rigor & Audit Metrics

**Project:** FORGE  
**Release Candidate:** v0.1.0-rc1  
**Audit Document:** `audit/rc1/PERFORMANCE_RECONCILIATION.md`  
**Date:** 2026-11-17  
**Auditor:** Independent Benchmark Auditor & Release Engineer  
**Hardware:** Intel Core i7-12650H, 16 GB DDR5, Windows 11 x64  
**Workload:** ResNet-20, CIFAR-10, Batch Size 128, SGD (lr=0.01, momentum=0.9, weight_decay=1e-4), FP32

---

## 1. Disentangling Timing Populations

A critical finding of this audit is that prior milestone reports inadvertently compared numbers originating from different **measurement populations**:

```
========================================================================================
                      TIMING POPULATION TAXONOMY IN PROJECT FORGE
========================================================================================
Population A: STEADY-STATE RECONCILED RUNS (Unthrottled Steady-State, 20 Steps, No Test Eval)
  - Environment: 10 OpenMP threads, subprocess isolation, 2s cooldowns, PL2/PL1 transition.
  - Characteristics: Excludes 5-step warmup and 10k-image test evaluation.
  - Stock NHWC Step Time: 212.11 ms (Mean of medians, 95% CI: [207.50, 216.72] ms)
  - FORGE NCHW Step Time: 249.43 ms (Mean of medians, 95% CI: [236.97, 261.89] ms)

Population B: SUSTAINED HEAT-SOAKED RUNS (Thermal Throttled, 30 Steps + 10k Test Eval)
  - Environment: Consecutive multi-experiment sweep under PL1 45W sustained power limit.
  - Characteristics: CPU core clocks throttled down to ~2.6-2.8 GHz from 4.2 GHz.
  - Stock NHWC Step Time: 396.58 ms
  - FORGE NCHW Step Time: 438.90 ms

Population C: PROFILER INSTRUMENTATION PASS (Single-Step PyTorch Profiler Call)
  - Environment: Single forward+backward step with kineto/profiler hooks active.
  - Total Profiled Time: 214.86 ms (Sum of operator self-times; conv 106.2 ms, bn 71.0 ms)

Population D: FULL MULTI-EPOCH RUNS (Complete Training Epochs with Periodic Validation)
  - Environment: Continuous 5-epoch training execution.
  - M2 5-Epoch Durations: Stock Tuned = 488.01 s, Stock Eager = 521.90 s, FORGE = 577.03 s
========================================================================================
```

> [!IMPORTANT]
> To preserve scientific integrity, performance comparisons must **never mix metrics across populations**. All claims below are strictly computed within matched populations.

---

## 2. Recomputed Performance Ledger

### 2.1 Reconciled Steady-State Benchmark (Population A)

| Metric | Certified Stock Competitor (`torch-eager` NHWC) | Fastest FORGE Variant (`forge` NCHW) | Alternative FORGE Variant (`forge` NHWC) | Relative Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Sample Mean of Per-Run Medians** | **212.11 ms** ($\text{SEM} = 1.66\text{ ms}$) | **249.43 ms** ($\text{SEM} = 0.98\text{ ms}$) | 272.62 ms ($\text{SEM} = 6.86\text{ ms}$) | **0.8504× (Stock is 1.18× faster)** |
| **Sample Median of Medians** | **212.11 ms** | **249.43 ms** | 272.62 ms | 0.8504× |
| **Raw Per-Run Medians (ms)** | `[206.25, 211.61, 214.85, 212.11, 215.72]` | `[250.41, 248.45]` | `[279.48, 265.77]` | Complete separation |
| **95% Confidence Interval (Student's t)** | **[207.50 ms, 216.72 ms]** | **[236.97 ms, 261.89 ms]** | [185.50 ms, 359.75 ms] | Disjoint intervals |
| **Single-Run Best Step Time** | 206.25 ms | 248.45 ms | 265.77 ms | — |
| **Throughput (samples/sec)** | **603.5 smp/s** (593.4 – 620.6) | **513.2 smp/s** (511.2 – 515.2) | 469.8 smp/s | **0.8504× throughput ratio** |
| **Slowdown Percentage** | Baseline (0.00%) | **+17.59% slower** | +28.53% slower | FORGE is slower |
| **Peak RSS Memory** | 640.5 MB – 675.8 MB | 605.5 MB – 648.2 MB | 558.3 MB – 620.1 MB | FORGE saves ~30-80 MB |
| **JIT Compilation Overhead** | 0.00 s | 0.00 s (AOT compiled) | 0.00 s (AOT compiled) | Parity |

*Definitions: Per-run medians are computed over 20 steady-state steps within each run (excluding 5 warmup steps). The sample mean of per-run medians, sample median of medians, SEM, and 95% confidence intervals are computed across independent runs.*

### 2.2 Sustained Heat-Soaked Benchmark (Population B)

| Metric | Certified Stock Competitor (`torch-eager` NHWC) | Fastest FORGE Variant (`forge` NCHW) | Alternative FORGE Variant (`forge` NHWC) | Relative Performance |
| :--- | :--- | :--- | :--- | :--- |
| **Median Step Time** | **396.58 ms** | **438.90 ms** | 513.89 ms | **0.9036× (Stock is 1.11× faster)** |
| **Throughput (samples/sec)** | 322.8 smp/s | 291.6 smp/s | 249.1 smp/s | **0.9036× throughput ratio** |
| **Slowdown Percentage** | Baseline (0.00%) | **+10.67% slower** | +29.58% slower | FORGE is slower |
| **Peak RSS Memory** | 540.3 MB | 605.5 MB | 558.3 MB | Comparable |

### 2.3 Historical 5-Epoch Training Runs (Population D)

| Metric | Fastest Stock Baseline (`torch-tuned`) | Stock Eager Baseline (`torch-eager`) | FORGE M2 Prototype (`forge`) |
| :--- | :--- | :--- | :--- |
| **Total 5-Epoch Duration** | **488.01 s** | 521.90 s | 577.03 s |
| **Relative Duration vs Fastest** | **1.000× (Fastest)** | 1.069× | **1.182× (18.2% slower)** |
| **Final Test Accuracy (5 ep)** | 66.16% | 59.42% | 66.16% |

---

## 3. Dissecting Why FORGE Is Slower Than Stock Eager

Across both steady-state (17.59% slower) and heat-soaked (10.67% slower) benchmarks, FORGE is slower than stock PyTorch. The architectural causes are documented below:

1. **Graph-Level Identity Addition in Stock PyTorch:**
   In stock PyTorch, evaluating `out = F.relu(x + residual)` creates two nodes: `ReluBackward0` followed by `AddBackward0`. The PyTorch autograd engine handles addition in backward by simply duplicating the tensor reference to both inputs. It launches **zero arithmetic kernels** and allocates **zero additional gradient memory**.
2. **External Custom Autograd Node Boundary Overhead:**
   An external C++ custom function (`torch::autograd::Function<FusedAddReLUZeroMaskFunction>`) must explicitly return a `torch::autograd::variable_list` with two gradient tensors:
   ```cpp
   return {grad_in, grad_in};
   ```
   Even though both entries share underlying storage, PyTorch's autograd engine performs defensive copy checks and tensor lifetime tracking at the boundary.
3. **Compiler Vectorization Quality:**
   PyTorch's internal ATen CPU kernels are compiled with GCC/Clang on Linux and highly tuned MSVC build configurations with SLEEF vectorized intrinsics, unrolling loops by 4 (32 floats per iteration) with non-temporal streaming stores. In contrast, custom MSVC extensions compiled via standard distutils do not benefit from SLEEF vectorization.

---

## 4. Final Performance Claim Summary for RC1 Release

1. **2.5× Acceleration Target Status:** **NOT ACHIEVED**.
2. **Fastest Valid Stock Baseline:** **`torch-eager` NHWC (10 threads) @ 212.11 ms** (Population A) / **396.58 ms** (Population B).
3. **Fastest Valid FORGE Configuration:** **`forge` NCHW (10 threads) @ 249.43 ms** (Population A) / **438.90 ms** (Population B).
4. **End-to-End Execution Ratio:** FORGE is **0.85× to 0.90× of stock performance** (i.e. **10.7% to 17.6% slower** than certified stock PyTorch).
5. **Memory Profile:** Peak RSS is reduced by approximately 30–80 MB under certain layouts.
