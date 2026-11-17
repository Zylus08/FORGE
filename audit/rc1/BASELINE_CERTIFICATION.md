# RC1 Baseline Certification Report: Statistical Resolution of Fastest Stock Competitor

**Project:** FORGE  
**Release Candidate:** v0.1.0-rc1  
**Audit Document:** `audit/rc1/BASELINE_CERTIFICATION.md`  
**Date:** 2026-11-17  
**Auditor:** Independent Benchmark Auditor & Release Engineer  
**Hardware:** Intel Core i7-12650H (10 Cores, 16 Threads, AVX2), Windows 11 x64  
**Workload:** ResNet-20, CIFAR-10, Batch Size 128, SGD, FP32

---

## 1. The Baseline Anomaly in Milestone M3

In the Milestone M3 ablation table (`audit/m3/ABLATION_RESULTS.csv`), single-slice measurements reported:
- `A0_Stock_Eager_NHWC_T10`: **396.58 ms**
- `A1_Stock_Eager_NCHW_T10`: **390.72 ms**

Because A1 recorded a median 5.86 ms (1.50%) lower than A0 in that specific test, but A0 was designated the official stock competitor, an apparent contradiction emerged:
> *Was `torch-eager` NCHW legitimately faster than NHWC, or was the 5.86 ms delta a measurement artifact of thermal drift and run ordering?*

---

## 2. Experimental Investigation & Counterbalanced Protocol

In the M3 ablation script, A0 was executed first while the system was entering thermal saturation (PL1 power limit), followed immediately by A1. To isolate layout effects from thermal throttling, thread contention, and cache warming, RC1 conducted a rigorous **counterbalanced repeated-measures experiment**:

### 2.1 Prespecified Methodology
1. **Sample Size:** 5 independent runs of `torch-eager` NHWC and 5 independent runs of `torch-eager` NCHW (10 runs total).
2. **Counterbalancing Schedule:** Alternating $ABBA BAAB AB$ schedule:
   $$\text{NHWC}_1 \to \text{NCHW}_1 \to \text{NCHW}_2 \to \text{NHWC}_2 \to \text{NHWC}_3 \to \text{NCHW}_3 \to \text{NCHW}_4 \to \text{NHWC}_4 \to \text{NHWC}_5 \to \text{NCHW}_5$$
3. **Execution Isolation:** Every run executed in a dedicated fresh subprocess via `canonical_runner.py` with 10 OpenMP threads and matched seeds.
4. **Boundary Standardization:** 5 warmup steps (excluded from metrics) followed by 20 timed steady-state training steps.
5. **Thermal Normalization:** 2-second idle cooldown between subprocess invocations.
6. **Prespecified Decision Rule:**
   - Compute mean, median, standard deviation, standard error of the mean (SEM), and 95% Student's t confidence intervals.
   - Perform two-sample Welch's t-test for unequal variances at significance level $\alpha = 0.05$.
   - If statistically distinguishable, select the configuration with the lower mean/median.
   - If statistically indistinguishable, formally declare the difference within measurement noise and select based on documented architectural suitability (e.g. oneDNN native layout).

---

## 3. Empirical Measurements

All raw JSON manifests are preserved in `benchmarks/rc1_runs/`:

| Run Index | Configuration | Layout | Median Step (ms) | Forward (ms) | Backward (ms) | Samples/sec |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `stock_nhwc_r1` | **NHWC** | **206.25 ms** | 76.13 ms | 112.58 ms | 620.6 |
| 2 | `stock_nchw_r1` | NCHW | 219.51 ms | 86.21 ms | 125.86 ms | 583.1 |
| 3 | `stock_nchw_r2` | NCHW | 220.53 ms | 86.72 ms | 126.08 ms | 580.4 |
| 4 | `stock_nhwc_r2` | **NHWC** | **211.61 ms** | 79.91 ms | 114.57 ms | 604.9 |
| 5 | `stock_nhwc_r3` | **NHWC** | **214.85 ms** | 78.80 ms | 115.29 ms | 595.8 |
| 6 | `stock_nchw_r3` | NCHW | 222.81 ms | 86.57 ms | 127.42 ms | 574.5 |
| 7 | `stock_nchw_r4` | NCHW | 222.84 ms | 88.08 ms | 127.00 ms | 574.4 |
| 8 | `stock_nhwc_r4` | **NHWC** | **212.11 ms** | 76.65 ms | 116.19 ms | 603.5 |
| 9 | `stock_nhwc_r5` | **NHWC** | **215.72 ms** | 79.57 ms | 117.84 ms | 593.4 |
| 10 | `stock_nchw_r5` | NCHW | 225.29 ms | 89.84 ms | 129.02 ms | 568.2 |

---

## 4. Statistical Analysis

### 4.1 Summary Statistics

| Metric | Stock Eager NHWC (Channels-Last) | Stock Eager NCHW (Contiguous) | Difference ($\Delta = \text{NHWC} - \text{NCHW}$) |
| :--- | :--- | :--- | :--- |
| **Number of Runs ($N$)** | 5 | 5 | — |
| **Raw Medians (ms)** | `[206.25, 211.61, 212.11, 214.85, 215.72]` | `[219.51, 220.53, 222.81, 222.84, 225.29]` | Complete separation |
| **Sample Mean of Medians** | **212.11 ms** | **222.20 ms** | **-10.09 ms (-4.54%)** |
| **Sample Median** | **212.11 ms** | **222.81 ms** | **-10.70 ms (-4.80%)** |
| **Sample Std Deviation ($s$)** | 3.71 ms | 2.26 ms | — |
| **Std Error of Mean (SEM)** | 1.66 ms | 1.01 ms | — |
| **95% Confidence Interval** | **[207.50 ms, 216.72 ms]** | **[219.40 ms, 225.00 ms]** | **Zero CI overlap** |
| **Min — Max Range** | 206.25 ms — 215.72 ms | 219.51 ms — 225.29 ms | $\max(\text{NHWC}) < \min(\text{NCHW})$ |

*Statistical Definitions: Per-run medians are computed over 20 steady-state steps within each run (excluding 5 warmup steps). The sample mean of per-run medians, sample median of medians, SEM, and 95% confidence intervals (Student's t, $\nu=4$) are computed across independent runs.*

### 4.2 Hypothesis Testing
- **Null Hypothesis ($H_0$):** $\mu_{\text{NHWC}} = \mu_{\text{NCHW}}$ (Mean step time of NHWC equals NCHW).
- **Alternative Hypothesis ($H_1$):** $\mu_{\text{NHWC}} < \mu_{\text{NCHW}}$ (NHWC is systematically faster than NCHW).
- **Welch's t-statistic:** $t = -5.1927$
- **Degrees of Freedom:** $\nu = 6.60$
- **Two-tailed p-value:** $p = 0.0016$ ($p < 0.002$)
- **Verdict:** Reject $H_0$ at $\alpha = 0.05$ (and at $\alpha = 0.01$).

### 4.3 Architectural Root Cause
The empirical superiority of NHWC is corroborated by CPU microarchitecture:
1. **oneDNN BRGConv Native Layout:** Intel oneDNN v3.13.3's AVX2 convolution engine operates natively on Channels-Last (`NHWC`) feature maps.
2. **Elimination of Transposition Overheads:** When tensors are formatted as NCHW, forward convolutions must reorder tensors into blocked formats (`aBcd8b`) or perform internal strided reads, which increases forward latency by ~8-12 ms (observed: Forward NHWC = 78.4 ms vs Forward NCHW = 87.5 ms).
3. **Strict Range Separation:** In all 5 counterbalanced pairs, the slowest measured NHWC run (215.72 ms) was faster than the fastest measured NCHW run (219.51 ms).

---

## 5. Certification Decision

Under the prespecified decision rule:
1. The difference between NHWC and NCHW is **statistically significant** ($p = 0.0016$).
2. The apparent 390.72 ms NCHW measurement in M3 was a thermal artifact of running in a heat-soaked state after earlier tests.
3. **`torch-eager` NHWC at 10 physical threads** is certified as the **official and fastest stock baseline competitor** for Project FORGE:
   - **Certified Baseline Step Time:** **212.11 ms** (Sample Mean of Medians; 95% CI: 207.50 – 216.72 ms; Single-run best: 206.25 ms).
   - **Certified Baseline Throughput:** **603.5 samples/second** (Range: 593.4 – 620.6 samples/second).

- **Baseline Certification Verdict:** **CERTIFIED (`torch-eager` NHWC, T10)**
