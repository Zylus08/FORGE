# RC1 Amdahl Analysis Correction: Reconciled Timing Boundaries & Theoretical Bounds

**Project:** FORGE  
**Release Candidate:** v0.1.0-rc1  
**Audit Document:** `audit/rc1/AMDAHL_CORRECTION.md`  
**Date:** 2026-11-17  
**Auditor:** Independent Benchmark Auditor & Release Engineer  
**Workload:** ResNet-20, CIFAR-10, Batch Size 128, SGD, FP32  
**Platform:** Intel Core i7-12650H, 10 Cores / 16 Threads, AVX2

---

## 1. Discrepancy Forensic: Profiler vs Benchmark Population

Milestone M3 identified a discrepancy between two timing figures:
- The operator profiler breakdown reported a total step duration of **214.86 ms**.
- The canonical benchmark runner reported step durations between **390 ms and 514 ms**.

### Root Cause Analysis
The two figures represent distinct physical operating conditions:
1. **Profiler Breakdown (214.86 ms):** Measured on an isolated, single-step training pass before sustained multi-run heating. This closely matches our counterbalanced Population A steady-state measurement (**212.11 ms**). Under this unthrottled state, the CPU operates near maximum PL2 burst clocks (~4.2–4.5 GHz).
2. **Multi-Run Benchmark Duration (390–514 ms):** Measured during continuous back-to-back ablation experiments that included repeated CIFAR-10 test set evaluations (10,000 images per epoch). Under continuous multi-thread load, the Intel Core i7-12650H enters its PL1 sustained thermal envelope (45W TDP), throttling core clocks down to ~2.6–2.8 GHz (a ~35–40% clock frequency reduction).

> [!NOTE]
> Amdahl's Law calculations must be performed using **internally consistent timing boundaries**. Mixing unthrottled profiler percentages with thermally throttled multi-run totals creates mathematical distortion. All derivations below are established within the verified 212–215 ms steady-state baseline population.

---

## 2. Internally Consistent Operator Decomposition

From the verified steady-state profiler audit (`audit/m3/BOTTLENECK_PROFILE.md`), normalized to total step time $T = 214.86\text{ ms}$:

| Domain ($i$) | Included Operators | Time ($T_i$) | Workload Share ($f_i$) |
| :--- | :--- | :--- | :--- |
| **Convolutions** | `aten::convolution`, `aten::mkldnn_convolution`, `convolution_backward` | 106.21 ms | **49.43%** |
| **Batch Normalization** | `aten::native_batch_norm`, `native_batch_norm_backward` | 71.01 ms | **33.05%** |
| **Residual Add + ReLU** | `aten::relu`, `aten::add`, `aten::threshold_backward` | 18.91 ms | **8.80%** |
| **Linear + Pooling + Loss** | `aten::linear`, `aten::avg_pool2d`, `aten::cross_entropy_loss` | 12.33 ms | **5.74%** |
| **SGD Optimizer** | `optimizer.step()` (momentum updates, weight decay) | 6.40 ms | **2.98%** |
| **Total Baseline Step** | | **214.86 ms** | **100.00%** |

---

## 3. Four-Tiered Amdahl Feasibility Analysis

Let $S$ represent the overall speedup factor:
$$S = \frac{T_{\text{baseline}}}{T_{\text{accelerated}}} = \frac{1}{\sum_{i} \frac{f_i}{k_i}}$$
where $k_i$ is the speedup factor achieved on component $i$.

### Tier 1: Fixed-Convolution Upper Bounds ($k_{\text{conv}} = 1.0$)
If convolutions are left unmodified ($k_{\text{conv}} = 1.0$, $f_{\text{conv}} = 0.4943$):

1. **Residual Fusion Only ($k_{\text{add\_relu}} \to \infty$, all others $k = 1.0$):**
   $$S_{\max} = \frac{1}{(1 - 0.0880) + \frac{0.0880}{\infty}} = \frac{1}{0.9120} \approx \mathbf{1.0965\times}$$
   *Maximum theoretical speedup from instantaneous residual Add+ReLU fusion is **9.65%**.*

2. **All Non-Convolution Operations Accelerated to Zero ($k_{\text{non\_conv}} \to \infty$):**
   Even if BatchNorm, Add, ReLU, Linear, Pooling, Loss, and Optimizer updates executed in $0.00\text{ ms}$:
   $$S_{\max} = \frac{1}{f_{\text{conv}}} = \frac{1}{0.4943} \approx \mathbf{2.023\times}$$
   *Therefore, without accelerating convolutions, the 2.50× target cannot be reached under any circumstances.*

---

### Tier 2: Hypothetical Convolution Acceleration Scenarios
To reach $S = 2.50\times$, total step time must reach $T_{\text{target}} = \frac{214.86\text{ ms}}{2.50} = \mathbf{85.94\text{ ms}}$.

If non-convolution operations remain completely unoptimized ($T_{\text{non\_conv}} = 108.65\text{ ms}$):
$$T_{\text{target}} - T_{\text{non\_conv}} = 85.94\text{ ms} - 108.65\text{ ms} = -22.71\text{ ms}$$
Because non-convolution time alone exceeds the entire target budget, **accelerating convolutions in isolation cannot achieve 2.5×**.

If non-convolutions are partially optimized by factor $k_{\text{non\_conv}}$:
$$T_{\text{conv\_budget}} = 85.94\text{ ms} - \frac{108.65\text{ ms}}{k_{\text{non\_conv}}}$$

The required convolution speedup $k_{\text{conv}} = \frac{106.21\text{ ms}}{T_{\text{conv\_budget}}}$ evaluates as follows:

| Non-Conv Reduction | Non-Conv Time | Budget Left for Convolutions | Required Convolution Speedup ($k_{\text{conv}}$) | Feasibility Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **0%** ($k=1.00$) | 108.65 ms | Negative (-22.71 ms) | Not Possible | Budget exceeded by non-conv |
| **30%** ($k=1.43$) | 76.06 ms | 9.88 ms | **10.75×** | Unrealistic in FP32 on CPU |
| **50%** ($k=2.00$) | 54.33 ms | 31.61 ms | **3.36×** | Requires major algorithmic shift |
| **70%** ($k=3.33$) | 32.60 ms | 53.34 ms | **1.99×** | Requires co-optimization of all paths |

---

### Tier 3: Hypothetical Combined System Co-Optimization
To achieve 2.5× end-to-end acceleration under FP32, an engineering architecture would need:
1. **Convolution Acceleration:** ~2.0× to 3.5× via direct Winograd filtering, low-overhead micro-GEMM JIT kernels, or epilogue fusion.
2. **BatchNorm Acceleration:** ~2.5× via in-place single-pass forward/backward kernels that eliminate memory passes.
3. **Activation & Residual Fusion:** ~1.5× via register-resident post-ops directly within GEMM epilogues.
4. **Graph-Level Autograd Routing:** Eliminating all custom function boundary copies.

---

### Tier 4: Actual Measured Acceleration in FORGE
- **Target:** 2.50× speedup ($T \le 85.94\text{ ms}$).
- **Measured Stock Baseline:** 212.11 ms (steady-state) / 396.58 ms (heat-soaked).
- **Measured FORGE:** 249.43 ms (steady-state) / 438.90 ms (heat-soaked).
- **Actual Measured Ratio:** **0.8504× to 0.9036×** (+10.7% to +17.6% slower).

---

## 4. Retraction of Unsupported Claims

In accordance with scientific audit standards, the following prior assertions from Milestone M3 are explicitly revised:
1. **Retraction of "Physically Impossible" Assertion:** The claim that 2.5× is "physically impossible in nature" is formally retracted. While 2.5× is unachieved within the current software architecture and bounded by Amdahl's Law when targeting only activations, it is not a violation of physical laws if full-stack co-optimization (including convolutions) is undertaken.
2. **Correction of "Mandatory 3× Convolution Speedup":** As demonstrated in Tier 2, the required convolution speedup is a function of non-convolution optimization. If non-convolutions are reduced by 70%, the required convolution speedup is 1.99×, not 3.36×.
3. **Audited Scientific Conclusion:** Residual Add+ReLU fusion alone has a theoretical speedup ceiling of **1.0965×**. Because Project FORGE M1–M3 focused primarily on activation and residual fusion while leaving oneDNN convolutions unaccelerated, the project could not mathematically achieve its 2.50× target.
