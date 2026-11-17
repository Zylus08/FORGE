# RC1 Accuracy Claims Audit: Tested Horizons, Empirical Evidence & Unverified Frontiers

**Project:** FORGE  
**Release Candidate:** v0.1.0-rc1  
**Audit Document:** `audit/rc1/ACCURACY_CLAIMS.md`  
**Date:** 2026-11-17  
**Auditor:** Independent Benchmark Auditor & Release Engineer  
**Workload Contract:** ResNet-20 (269,722 trainable parameters), CIFAR-10, Batch Size 128, SGD (lr=0.01, momentum=0.9, weight_decay=1e-4), Target: $\ge 85\%$ Top-1 Accuracy at 160 Epochs

---

## 1. Executive Summary & Statement of Correction

The frozen workload contract specifies that the training pipeline must achieve at least **85.0% top-1 validation accuracy** at the designated 160-epoch horizon.

In prior milestone reports, it was asserted that because FORGE achieved bitwise numerical parity with stock PyTorch over short training windows ($L_\infty < 10^{-6}$ over 10 steps), the 85% accuracy threshold at 160 epochs was "guaranteed" or "verified."

> [!WARNING]
> This claim is scientifically invalid and is **formally corrected**.
> Bitwise or near-bitwise parity over short training horizons confirms that the software stack faithfully reproduces the initial gradient trajectory. However, it **does not constitute empirical proof of 160-epoch convergence**.
> Long-horizon training can encounter cumulative numerical drift, gradient explosion/vanishing across learning-rate decay boundaries (epochs 80 and 120), or multithreaded non-deterministic sample accumulation.

---

## 2. Actual Tested Horizons & Observed Evidence

The table below documents every training horizon actually executed and verified across Project FORGE milestones:

| Horizon Tested | Total Steps | Execution Date | Stock Baseline Accuracy | FORGE Accuracy | Verified Properties |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1 Step** | 1 step | 2026-11-16 (M1) | N/A | N/A | Operator forward/backward output parity |
| **10 Steps** | 10 steps | 2026-11-17 (M3/RC1) | N/A (Pseudo-batch) | N/A (Pseudo-batch) | Bitwise gradient and parameter parity ($L_\infty < 10^{-6}$) |
| **20 Steps** | 20 steps | 2026-11-17 (RC1) | Train: 16.62% | Train: 16.19% | Subprocess-isolated steady-state step timing |
| **30 Steps** | 30 steps | 2026-11-17 (M3) | Test: 25.45% | Test: 20.99% | Early epoch 1 convergence dynamics |
| **5 Epochs** | 1,955 steps | 2026-11-16 (M2) | Test: 66.16% (`tuned`) | Test: 66.16% (`forge`) | Multi-epoch convergence parity |
| **160 Epochs** | 62,560 steps | **UNEXECUTED** | **UNEXECUTED** | **UNEXECUTED** | Target horizon remains unexecuted |

---

## 3. Why the 160-Epoch Horizon Was Not Executed

Executing the full 160-epoch training horizon was explicitly deferred in accordance with experimental safety and recovery constraints:

1. **Measured Execution Duration:**
   - Steps per epoch: $\lceil 50,000 / 128 \rceil = 391\text{ steps/epoch}$.
   - Measured step time (sustained): $\approx 396.58\text{ ms}$.
   - Duration per epoch: $391 \times 0.39658\text{ s} \approx 155.1\text{ s} \quad (\approx 2.58\text{ minutes})$.
   - Evaluation per epoch (10k test images): $\approx 6.0\text{ s}$.
   - Total epoch time: $\approx 161.1\text{ s}$.
   - **Total 160-Epoch Duration:**
     $$160 \times 161.1\text{ s} = 25,776\text{ s} \approx \mathbf{7.16\text{ hours}}$$
2. **Thermal & Infrastructure Risk:**
   Running continuous 100% 10-core CPU training for 7.16 hours on a laptop chassis risks extreme thermal throttling (reducing clocks to ~2.0 GHz, further extending execution to 9+ hours), IPC worker deadlocks, or task timeouts.
3. **Audit Policy:**
   Milestone M3 and RC1 instructions strictly mandate bounded verification experiments and prohibit launching unmonitored multi-hour training runs.

---

## 4. What Is Verified vs What Remains Unverified

### 4.1 Verified Properties
- **Parameter Count:** The model contains exactly **269,722 trainable parameters**, matching standard ResNet-20 for CIFAR-10.
- **Operator Numerical Equivalence:** Over $3.67 \times 10^6$ activation elements, FORGE M3 Zero-Mask achieves Max $|Diff|_\infty = 0.000000$ vs stock PyTorch reference.
- **Short-Horizon Optimizer Step Equivalence:** Over 10 deterministic steps, weights, gradients, momentum buffers, and running batch-norm statistics maintain $L_\infty < 10^{-6}$.
- **5-Epoch Convergence:** Milestone M2 verified that after 5 complete epochs, FORGE achieves 66.16% test accuracy, matching tuned stock PyTorch.

### 4.2 What Remains Unverified
- **160-Epoch Final Accuracy:** Whether FORGE reaches $\ge 85.0\%$ validation accuracy at epoch 160 remains **empirically unverified**.
- **Learning Rate Schedule Discontinuities:** Behavior across the 10× learning rate drops at epoch 80 and epoch 120 has not been observed.
- **Cumulative Floating-Point Drift:** Over 62,560 stochastic optimization steps, non-associativity of floating-point SIMD reductions could theoretically cause minor divergence from a stock baseline run.

---

## 5. Formal Accuracy Audit Classification

- **Target Workload Accuracy (160 Epochs):** $\ge 85.0\%$ Top-1 Accuracy.
- **Observed Accuracy (5 Epochs):** $66.16\%$.
- **Short-Horizon Trajectory Parity:** **PASS ($L_\infty < 10^{-6}$)**.
- **Full-Horizon Verification Status:** **UNVERIFIED AT FULL HORIZON (MATHEMATICALLY PLAUSIBLE, EMPIRICALLY UNEXECUTED DUE TO 7.16-HOUR RUNTIME BOUNDARY)**.
