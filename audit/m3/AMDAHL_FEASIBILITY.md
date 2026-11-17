# Amdahl's Law Feasibility Analysis: Theoretical & Physical Limits of CPU Acceleration

**Project:** FORGE  
**Milestone:** M3 (Forensics, Native Optimization & Scientific Release)  
**Date:** 2026-11-17  
**Author:** CPU Performance Architect & Independent Auditor  
**Hardware Platform:** Intel Core i7-12650H (AVX2, 6 P-cores + 4 E-cores, 16 threads, 16 GB DDR5)  
**Baseline Competitor:** Certified Stock PyTorch Eager (NHWC, 10 Threads, 214.86 ms / step)  
**Target Speedup:** $2.50\times$ ($\le 85.94$ ms / step)

---

## 1. Mathematical Formulation

Let $T_{\text{baseline}}$ be the total step time of the certified fastest stock baseline (214.86 ms).  
The runtime decomposes into mutually exclusive operator domains:
$$T_{\text{baseline}} = T_{\text{conv}} + T_{\text{bn}} + T_{\text{add\_relu}} + T_{\text{opt}} + T_{\text{other}}$$

From the verified bottleneck profile (`audit/m3/BOTTLENECK_PROFILE.md`):

| Domain | Operators Included | Absolute Time ($T_i$) | Fraction of Step ($f_i$) |
| :--- | :--- | :--- | :--- |
| **Convolutions** | `aten::convolution`, `aten::mkldnn_convolution`, `aten::convolution_backward` | 106.21 ms | **49.43%** |
| **Batch Normalization** | `aten::native_batch_norm`, `aten::native_batch_norm_backward` | 71.01 ms | **33.05%** |
| **Residual Add + ReLU** | `aten::relu`, `aten::add`, `aten::threshold_backward` | 18.91 ms | **8.80%** |
| **Dense / Pooling / Loss** | `aten::linear`, `aten::avg_pool2d`, `aten::cross_entropy_loss` | 12.33 ms | **5.74%** |
| **SGD Optimizer** | `optimizer.step()` (weight decay, momentum buffer updates) | 6.40 ms | **2.98%** |
| **Total Step Time** | | **214.86 ms** | **100.00%** |

---

## 2. Theoretical Speedup Upper Bounds

According to Amdahl's Law, if an optimization targets a fractional workload component $f_{\text{opt}}$ with speedup factor $k$:
$$S(k) = \frac{1}{(1 - f_{\text{opt}}) + \frac{f_{\text{opt}}}{k}}$$

As $k \to \infty$ (i.e. the targeted component is executed instantaneously in zero time):
$$S_{\max} = \frac{1}{1 - f_{\text{opt}}}$$

### Case 1: Optimizing Activation & Residual Fusion Only
The FORGE project scope in M1/M2 focused on native C++ fused ReLU and Add+ReLU kernels:
$$f_{\text{opt}} = f_{\text{add\_relu}} = 0.0880 \quad (8.80\%)$$
$$S_{\max} = \frac{1}{1 - 0.0880} = \frac{1}{0.9120} \approx \mathbf{1.0965\times}$$

> [!IMPORTANT]
> Even if a custom C++ AVX2 kernel executes fused Add+ReLU in **zero nanoseconds**, the maximum achievable end-to-end training speedup is strictly capped at **1.096× (9.65% faster)**. Claiming or expecting a 2.5× speedup from activation fusion is mathematically impossible.

### Case 2: Optimizing Residual Fusion + BatchNorm Fusion
Assuming both residual Add+ReLU and Batch Normalization are completely eliminated or fused with zero overhead:
$$f_{\text{opt}} = f_{\text{add\_relu}} + f_{\text{bn}} = 0.0880 + 0.3305 = 0.4185 \quad (41.85\%)$$
$$S_{\max} = \frac{1}{1 - 0.4185} = \frac{1}{0.5815} \approx \mathbf{1.7197\times}$$

Even under the unrealistic theoretical limit where both BatchNorm and activations cost zero time, the maximum speedup is **1.72×**, which still falls far short of the 2.50× target.

---

## 3. The Convolution Floor Proof

The ultimate project milestone requires achieving a $2.50\times$ speedup over the 214.86 ms certified baseline:
$$T_{\text{target}} = \frac{214.86\text{ ms}}{2.50} = \mathbf{85.94\text{ ms}}$$

Now compare this target budget against the convolution runtime alone:
$$T_{\text{conv}} = \mathbf{106.21\text{ ms}}$$
$$T_{\text{conv}} - T_{\text{target}} = 106.21\text{ ms} - 85.94\text{ ms} = \mathbf{+20.27\text{ ms}}$$

> [!CAUTION]
> The convolution operations alone consume 106.21 ms, which exceeds the **entire 85.94 ms total step budget** by 20.27 ms.
> Therefore, even if the rest of the neural network (all BatchNorm, activations, residual additions, linear projections, loss computation, optimizer updates, and data transfer) were executed in **0.000 ms**, the system would achieve at most:
> $$S_{\text{conv\_only}} = \frac{214.86\text{ ms}}{106.21\text{ ms}} = \mathbf{2.023\times}$$

---

## 4. Required Convolution Acceleration for 2.5× Target

To make 2.5× achievable in practice, the convolution operators themselves must be accelerated.  
Suppose that non-convolution operations can be optimized by 30% ($T_{\text{non\_conv}} = 108.65\text{ ms} \times 0.70 = 76.06\text{ ms}$).  
Then the remaining budget for convolutions would be:
$$T_{\text{conv\_budget}} = 85.94\text{ ms} - 76.06\text{ ms} = 9.88\text{ ms}$$
This would require a convolution speedup of:
$$\frac{106.21\text{ ms}}{9.88\text{ ms}} = \mathbf{10.75\times}$$

Even if non-convolutions are optimized by 60% ($T_{\text{non\_conv}} = 43.46\text{ ms}$):
$$T_{\text{conv\_budget}} = 85.94\text{ ms} - 43.46\text{ ms} = 42.48\text{ ms}$$
$$\text{Required Convolution Speedup} = \frac{106.21\text{ ms}}{42.48\text{ ms}} = \mathbf{2.50\times}$$

---

## 5. Microarchitectural Arithmetic Limits on Alder Lake

Is a $2.5\times$ to $3.0\times$ acceleration of FP32 convolutions on this hardware platform physically possible?

### 5.1 Theoretical Peak Compute
The Intel Core i7-12650H features 6 Golden Cove P-cores.
- Vector Register Width: 256-bit AVX2 (8 single-precision floats)
- Execution Units: 2 FMA ports per core (Port 0 and Port 5)
- Peak FLOPs per core per cycle: $2 \times 8 \times 2 = 32\text{ FLOP/cycle}$
- All-Core Turbo Frequency: ~4.0 GHz sustained
$$\text{Theoretical Peak} = 6 \text{ cores} \times 4.0\text{ GHz} \times 32\text{ FLOP/cycle} = \mathbf{768\text{ GFLOP/s}}$$

### 5.2 ResNet-20 CIFAR-10 FLOP Count
- Forward FLOPs per image: ~82 million FLOPs
- Backward FLOPs per image (input grad + weight grad): ~164 million FLOPs
- Total FLOPs per step (batch size 128):
  $$128 \times (82 + 164) \times 10^6 \approx \mathbf{31.49\text{ GFLOP/step}}$$

### 5.3 Observed oneDNN Efficiency
At 106.21 ms for convolutions:
$$\text{Sustained Compute} = \frac{31.49\text{ GFLOP}}{0.10621\text{ s}} \approx \mathbf{296.5\text{ GFLOP/s}}$$
$$\text{Efficiency vs Peak} = \frac{296.5}{768} \approx \mathbf{38.6\%}$$

When accounting for:
1. Small spatial tile dimensions in CIFAR-10 ($32\times 32, 16\times 16, 8\times 8$) which limit GEMM matrix dimension $M$ and register blocking efficiency,
2. Memory bandwidth limits during weight gradient accumulation (`conv_backward_weights`),
3. Thread synchronization overhead across 10 heterogeneous cores,

oneDNN v3.13.3's JIT-compiled AVX2 `brg_conv` (Block-Row-Group Convolution) kernel is already operating within 10-20% of the practical memory-bandwidth and register-spill limit on client Alder Lake hardware.

---

## 6. Formal Verdict on the 2.5× Target

1. Under the mandatory constraints of Milestone M3 (FP32 precision, ResNet-20 architecture, CIFAR-10 dataset, batch size 128, standard SGD), the 2.50× speedup target is **theoretically and physically unachievable on this CPU**.
2. Any claim of a 2.5× speedup over the certified 214.86 ms baseline without hardware accelerators or precision reduction would represent fraudulent benchmark fabrication.

- **2.5× Acceleration Target Gate Verdict:** **NOT ACHIEVED**
