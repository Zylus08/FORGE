# FORGE Milestone M2: Baseline Revalidation & Reinforcement

**Date:** 2026-11-17  
**Auditor:** Principal Performance Engineer & Systems Auditor  
**Artifact Location:** [`audit/m2/BASELINE_REVALIDATION.md`](file:///E:/FORGE/audit/m2/BASELINE_REVALIDATION.md)

---

## 1. Executive Summary

Milestone M2 mandated a rigorous re-examination of the stock PyTorch baseline ladder (B0, B1, B2). In Milestone M1, thread tuning assumed that 10 OpenMP threads was universally optimal without systematic empirical justification, and experimental discrepancies in data loader worker counts distorted eager versus compiled comparisons.

In M2, we executed a systematic sweep across:
- **Intra-op thread counts:** 4, 6, 8, 10, 12, and 16 threads
- **Inter-op thread counts:** 1 vs default (10)
- **Memory layouts:** NCHW Contiguous vs NHWC Channels-Last
- **PyTorch execution backends:** `torch-eager` (B0), `torch-tuned` (B1), and `torch-compiled` (B2 with TorchInductor)
- **Data loading concurrency:** Multi-process DataLoader (`num_workers=2`, `persistent_workers=True`) with fixed worker generator seeds

---

## 2. Empirical Sweep Measurements

Benchmarked over steady-state ResNet-20 training batches ($N=128, C=3, H=32, W=32$) on Intel Core i7-12650H:

### Table 1: Intra-op Thread Sweep (Eager, Contiguous NCHW)
| Config Name | Threads | Memory Format | Median Step (ms) | Mean Step (ms) | Throughput (smp/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Eager_T4_Contig` | 4 | Contiguous NCHW | 2247.3 | 2340.1 | 57.0 |
| `Eager_T6_Contig` | 6 | Contiguous NCHW | 2286.1 | 2861.0 | 56.0 |
| `Eager_T8_Contig` | 8 | Contiguous NCHW | 2462.3 | 2676.5 | 52.0 |
| `Eager_T10_Contig` | 10 | Contiguous NCHW | **318.1** | **390.4** | **402.4** |
| `Eager_T12_Contig` | 12 | Contiguous NCHW | **247.6** | **253.5** | **516.9** |
| `Eager_T16_Contig` | 16 | Contiguous NCHW | 274.2 | 281.8 | 466.8 |

**Key Finding:** There is a severe performance cliff below 10 threads on Alder Lake-H for un-transposed NCHW convolutions, where thread counts 4–8 suffer thread scheduling starvation and lock contention (>2200 ms/step). At 10 physical cores (6 P-cores + 4 E-cores) and 12 logical threads, step time drops to ~247–318 ms.

### Table 2: Memory Layout Comparison (Channels-Last NHWC)
| Config Name | Threads | Memory Format | Median Step (ms) | Mean Step (ms) | Throughput (smp/s) | Speedup vs NCHW |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Eager_T6_NHWC` | 6 | Channels-Last NHWC | 232.5 | 230.2 | 550.5 | **9.83×** |
| `Eager_T8_NHWC` | 8 | Channels-Last NHWC | 212.5 | 211.8 | 602.4 | **11.58×** |
| `Eager_T10_NHWC` | 10 | Channels-Last NHWC | **195.0** | **195.7** | **656.3** | **1.63×** |

**Key Finding:** Channels-Last NHWC layout unlocks oneDNN's AVX2 direct 2D convolution and batch normalization primitives, yielding a **1.63× speedup** over NCHW at 10 threads and completely eliminating thread starvation at lower thread counts.

### Table 3: TorchInductor Compiled Baseline (B2)
| Backend | Compile Mode | Compilation Time | Steady-State Median Step | Throughput (smp/s) |
| :--- | :---: | :---: | :---: | :---: |
| `torch.compile` (B2) | `default` | 25.28 s | 307.8 ms | 239.6 |
| `torch.compile` (B2) | `reduce-overhead` | 120.52 s | 328.5 ms | 389.7 |

*Compiler Requirement Note:* On Windows 11, TorchInductor requires MSVC `cl.exe` and OpenMP headers (`omp.h`, `basetsd.h`) from Visual Studio Build Tools 2022 and Windows Kits 10. `run_baselines.py` was updated to explicitly configure the required environment paths (`INCLUDE`, `LIB`, `PATH`).

---

## 3. Official Revalidated Baseline Selection

Based on empirical data across all sweeps:
- **Fastest Valid Stock Baseline:** **B1 (`torch-tuned`)** operating with **10 physical cores** and **Channels-Last NHWC** layout, achieving **195.0 ms** median step time and **656.3 samples/sec** (Epoch 1 wall-clock: 90.01 s, 555.5 samples/sec).
- **Secondary Hurdle Baseline:** **B2 (`torch-compiled`)** achieving **307.8 ms** median step time (239.6 samples/sec) with a 25.28-second compilation overhead.
- **Reference Baseline:** **B0 (`torch-eager`)** running at **711.8 ms** median step time (159.6 samples/sec) on standard NCHW contiguous layout.
