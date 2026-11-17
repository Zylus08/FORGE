# FORGE Milestone M3: Canonical Benchmark Protocol & Methodology

**Project:** FORGE (PyTorch CPU Training Accelerator)  
**Workspace:** `E:\FORGE`  
**Tooling Implementation:** [`forge/benchmarking/canonical_runner.py`](file:///E:/FORGE/forge/benchmarking/canonical_runner.py)  
**Auditor Roles:** Principal CPU Performance Architect, Benchmark Auditor  
**Date:** 2026-11-17  
**Artifact Location:** [`audit/m3/CANONICAL_BENCHMARK_PROTOCOL.md`](file:///E:/FORGE/audit/m3/CANONICAL_BENCHMARK_PROTOCOL.md)

---

## 1. Motivation: Eliminating Measurement Flaws

In Milestone M1 and M2, various benchmark inconsistencies distorted performance comparisons:
1. **IPC Worker Leakage:** Running repeated DataLoader instances in the same Python process created dangling background worker processes on Windows, causing pipe contention (`WaitReason: Executive`).
2. **Layout Inconsistency:** Stock tuned was executed in Channels-Last (NHWC) format, while FORGE was run in Contiguous (NCHW) format, creating an asymmetric ~1.6× performance difference unrelated to custom kernels.
3. **Warm-Up Measurement Pollution:** Measuring step 0 on un-warmed thread pools included memory allocation cold faults and oneDNN primitive generation.
4. **Main-Process RNG Contamination:** Running eager mode with `num_workers=0` consumed main-process Python RNG, while tuned mode with `num_workers=2` consumed worker RNG, causing divergent image augmentations.

To establish absolute scientific rigor, Milestone M3 enforces a **Canonical Benchmark Protocol**.

---

## 2. Canonical Measurement Rules & Enforcements

### Rule 1: Subprocess Process Isolation
Every benchmark configuration is executed in a dedicated, clean Python subprocess via `canonical_runner.py`. This guarantees:
- Fresh CPU thread pools (`torch.set_num_threads` applied before any parallel operator initializes).
- Zero residual PyTorch CPU allocator memory fragmentation.
- Fresh OS process memory space and clean MSVC dynamic link states.
- Exactly recorded OS Process ID, Git commit hash, and peak Resident Set Size (RSS).

### Rule 2: Matched Measurement Boundaries
- **Timing Interval:** `t_start` is recorded immediately after batch data is received; `t_end` is recorded after `optimizer.step()`.
- **Component Breakdowns:**
  - Forward Duration: `t_fwd_end - t_fwd_start` (encompassing forward model invocation and loss calculation).
  - Backward Duration: `t_bwd_end - t_bwd_start` (`loss.backward()` autograd traversal and gradient computation).
  - Optimizer Duration: `t_opt_end - t_opt_start` (`optimizer.step()` parameter update).
  - Data Duration: `t_data_end - t_data_start` (queue wait time for next DataLoader batch).
- **Warm-Up Exclusion:** The first 5 steps of Epoch 1 are strictly designated as warm-up and excluded from reported steady-state latency metrics.

### Rule 3: Symmetric Layouts & Hardware Bindings
- Competing backends must be compared under matched memory formats:
  - **Primary Ladder:** Channels-Last (NHWC) across all backends (`torch-tuned`, `torch-compiled`, `forge`).
  - **Diagnostic Ladder:** Contiguous (NCHW) across all backends to isolate layout sensitivity.
- Threads are fixed to **10 physical cores** (`torch.set_num_threads(10)`, `torch.set_num_interop_threads(1)`), eliminating Alder Lake hyperthread contention.

### Rule 4: Workload & Numerical Parity Contract
- **Architecture:** ResNet-20 (269,722 parameters).
- **Dataset:** CIFAR-10 (50,000 train, 10,000 test).
- **Batch Size:** 128 (391 steps per epoch).
- **Optimizer:** SGD(lr=0.01, momentum=0.9, weight_decay=1e-4).
- **Loss:** CrossEntropyLoss.
- **Precision:** FP32 strictly.
- **Evaluation:** Evaluated on all 10,000 test images using `evaluate()` after freeing gradient memory (`optimizer.zero_grad(set_to_none=True)`).

---

## 3. Runner Architecture

The canonical runner is implemented in [`forge/benchmarking/canonical_runner.py`](file:///E:/FORGE/forge/benchmarking/canonical_runner.py):
- Supports `--backend [torch-eager, torch-tuned, torch-compiled, forge]`.
- Supports `--layout [NHWC, NCHW]`.
- Supports `--threads [int]` and `--interop [int]`.
- Supports `--epochs [int]` and optional `--timed-steps [int]`.
- Generates structured JSON outputs containing full telemetry, system metadata, Git commit, step latency percentiles (median, mean, P95), and validation metrics.
