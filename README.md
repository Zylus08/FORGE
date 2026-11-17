# FORGE — PyTorch CPU Training Accelerator (`v0.1.0-rc1`)

> **Release Candidate v0.1.0-rc1** — Research-grade experimental PyTorch CPU acceleration framework with original AVX2 kernels, numerical verification, profiling, and reproducible benchmark infrastructure.

---

## 1. Project Overview & Scientific Positioning

FORGE is an experimental research framework designed to investigate CPU operator fusion and native execution acceleration for deep neural network training in PyTorch without altering model architecture, loss functions, or mathematical contracts.

In accordance with strict scientific integrity standards:
- **2.5× Acceleration Target Status:** **NOT ACHIEVED**.
- **Official Certified Stock Baseline Competitor:** **`torch-eager` with Channels-Last (`NHWC`) layout at 10 physical threads** (**212.11 ms** / step, 603.5 samples/sec).
- **Fastest Valid FORGE Configuration:** **`forge` with Contiguous (`NCHW`) layout at 10 physical threads** (**249.43 ms** / step, 513.2 samples/sec).
- **End-to-End Execution Ratio:** FORGE achieves **0.8504× of stock performance** (i.e. **17.59% slower** than stock PyTorch eager).
- **Amdahl's Law Bottleneck:** Detailed operator profiling reveals that 2D convolutions account for **49.43%** and Batch Normalization accounts for **33.05%** of execution time (**82.48% combined**). Residual Add+ReLU fusion alone accounts for **8.80%**, establishing a theoretical speedup upper bound of **1.0965×** when convolutions remain unaccelerated.

---

## 2. Core Architectural Components

1. **Native C++20 AVX2 Zero-Mask Kernels (`native/src/fused_ops_m2.cpp`):**
   - High-performance fused ReLU and residual Add+ReLU operations with AVX2 SIMD intrinsics.
   - Zero-mask architecture: Eliminates 2 MB boolean mask DRAM allocation per stage by evaluating the derivative condition $out > 0.0$ directly in vector registers during backpropagation.
   - Native C++ custom autograd functions (`torch::autograd::Function<FusedAddReLUZeroMaskFunction>`) eliminating Python dispatch overhead.
2. **Deterministic Parity & Edge-Case Verification Suite (`tests/numerical/`):**
   - 29 regression tests verifying forward/backward parity ($L_\infty = 0.0$ across 3,674,112 elements).
   - Analytical finite-difference gradient checks (`torch.autograd.gradcheck`).
   - Verified IEEE-754 semantics: subgradient at zero ($0.0$), signed zero ($+0.0$ vs $-0.0$), NaN clamping, and in-place mutation detection.
   - Deterministic paired training trajectory verification over 10 steps ($L_\infty < 10^{-6}$ across all 269,722 parameters, momentum buffers, and running statistics).
3. **Subprocess-Isolated Canonical Benchmark Suite (`forge/benchmarking/canonical_runner.py`):**
   - Isolated subprocess execution preventing memory fragmentation, thread pool corruption, and thermal distortion.
   - Matched measurement boundaries excluding warmup steps, with Drive E: scratch directory protection.

---

## 3. Audited Benchmark Ledger (RC1 Certified)

All benchmarks measured on Intel Core i7-12650H (10 physical cores: 6 P-cores + 4 E-cores, 16 threads, AVX2, Windows 11 x64, 16 GB DDR5), frozen ResNet-20 CIFAR-10 FP32 workload, batch size 128:

| Configuration | Backend | Layout | Threads | Median Step Time (ms) | 95% Confidence Interval | Throughput (samples/s) | Relative Speedup | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`stock_nhwc`** | `torch-eager` | **NHWC** | **10** | **212.11 ms** | **[207.50, 216.72] ms** | **603.5** | **1.0000×** | **CERTIFIED STOCK BASELINE** |
| `stock_nchw` | `torch-eager` | NCHW | 10 | 222.20 ms | [219.40, 225.00] ms | 576.1 | 0.9546× | Valid Stock Eager |
| `stock_tuned` | `torch-tuned` | NHWC | 10 | 254.88 ms | [248.10, 261.66] ms | 502.2 | 0.8322× | Stock Tuned |
| **`forge_nchw`** | `forge` | **NCHW** | **10** | **249.43 ms** | **[236.97, 261.89] ms** | **513.2** | **0.8504×** | **FASTEST FORGE CONFIG** |
| `forge_nhwc` | `forge` | NHWC | 10 | 272.62 ms | [185.50, 359.75] ms | 469.8 | 0.7780× | FORGE Channels-Last |

*Statistical significance: Two-sample Welch's t-test confirms NHWC is faster than NCHW for stock PyTorch ($t = -5.1927, p = 0.0016$).*

---

## 4. Repository Structure

```
FORGE/
├── forge/                          # Python package
│   ├── backends/                   # Pluggable backends & native bindings
│   ├── benchmarking/               # Canonical subprocess benchmark runner
│   └── integration/                # PyTorch model wrappers & trainer hooks
├── native/                         # Native C++20 AVX2 kernel source
│   ├── include/                    # C++ extension header files
│   └── src/                        # Fused Add+ReLU & Zero-Mask implementations
├── workloads/                      # Frozen workload definitions (ResNet-20 / CIFAR-10)
├── tests/                          # 29 regression & numerical parity tests
│   ├── numerical/                  # Bitwise parity, gradcheck, paired training
│   ├── integration/                # Smoke training tests
│   ├── unit/                       # Model configuration & parameter count tests
│   └── benchmark/                  # Benchmark protocol tests
├── benchmarks/                     # Preserved raw JSON run manifests
│   ├── rc1_runs/                   # Counterbalanced RC1 baseline confirmation runs
│   └── m3_runs/                    # Milestone M3 ablation & tuning manifests
├── audit/                          # Scientific release audit documentation
│   ├── rc1/                        # RC1 Scientific Integrity & Baseline Certification
│   ├── m3/                         # Milestone M3 Forensic & Optimization Reports
│   ├── m2/                         # Milestone M2 Historical Reports
│   └── m1/                         # Milestone M1 Baseline Certification
├── tools/                          # Benchmark & reconciliation scripts
└── setup_native.py                 # PyTorch C++ extension build script
```

---

## 5. Quickstart & Verification

### 5.1 Rebuilding Native Extension
```powershell
$env:TMP = "E:\FORGE\.tmp"; $env:TEMP = "E:\FORGE\.tmp"
python setup_native.py build_ext --inplace
```

### 5.2 Running the Full Regression Suite
```powershell
python -m pytest tests/ -v
```

### 5.3 Executing the Canonical Benchmark
```powershell
# Certified stock competitor (NHWC, 10 threads)
python -m forge.benchmarking.canonical_runner --worker --backend torch-eager --layout NHWC --threads 10 --timed-steps 20 --skip-eval

# FORGE M3 Zero-Mask (NCHW, 10 threads)
python -m forge.benchmarking.canonical_runner --worker --backend forge --layout NCHW --threads 10 --timed-steps 20 --skip-eval
```

---

## 6. Audit & Scientific Integrity Documents

Complete forensic and experimental reports are available in [`audit/rc1/`](audit/rc1/):
- [`BASELINE_CERTIFICATION.md`](audit/rc1/BASELINE_CERTIFICATION.md): Statistical proof and certification of the fastest stock baseline.
- [`PERFORMANCE_RECONCILIATION.md`](audit/rc1/PERFORMANCE_RECONCILIATION.md): Recalculated metrics, population separation, and slowdown forensics.
- [`AMDAHL_CORRECTION.md`](audit/rc1/AMDAHL_CORRECTION.md): Internally consistent Amdahl derivations and theoretical speedup bounds.
- [`NUMERICAL_REVALIDATION.md`](audit/rc1/NUMERICAL_REVALIDATION.md): IEEE-754 semantics, zero subgradient, and bitwise parity.
- [`ACCURACY_CLAIMS.md`](audit/rc1/ACCURACY_CLAIMS.md): Disclosure of unexecuted 160-epoch horizon and verified short-horizon evidence.
- [`FINAL_RELEASE_DECISION.md`](audit/rc1/FINAL_RELEASE_DECISION.md): Formal dual gate verdicts and release candidate authorization.

---

## 7. License

MIT License — see [LICENSE](LICENSE).
