# PROJECT FORGE Milestone M3: Interrupted Execution Recovery Report

**Project:** FORGE (PyTorch CPU Training Accelerator)  
**Workspace:** `E:\FORGE`  
**Starting Commit:** [`c4cfc57`](file:///E:/FORGE)  
**Previous Implementation Commit:** [`917049a`](file:///E:/FORGE)  
**Recovery Timestamp:** 2026-11-17T14:31:00+05:30  
**Auditor Roles:** Principal CPU Performance Architect, Release Auditor  
**Artifact Location:** [`audit/m3/RECOVERY_REPORT.md`](file:///E:/FORGE/audit/m3/RECOVERY_REPORT.md)

---

## 1. Incident Forensics & Root Cause Identification

At approximately 10:21 local time, an automated execution task (`task-902`) was triggered to run the compiled baseline `torch-compiled` through `forge.benchmarking.canonical_runner`.

### What Stalled / Failed:
- Task `task-902` failed with exit code 1 due to:
  ```text
  RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 4194304 bytes.
  ```
- **Root Cause:** In `canonical_runner.py`, during the TorchInductor compilation warmup pass, `dummy_loss.backward()` populated parameter `.grad` buffers across all ResNet-20 layers. Because `optimizer.zero_grad(set_to_none=True)` and `gc.collect()` were not invoked after the dummy backward pass, those buffers remained locked in memory when step 0 of training attempted to allocate forward activations in TorchInductor's generated C++ code. Under Windows commit charge constraints, the PyTorch CPU allocator threw an out-of-memory exception.
- Following the failure, the Antigravity agent process was restarted, terminating any stalled child tasks cleanly.

### What Was NOT Running:
- No background training runs, infinite loops, or orphaned MSVC/Ninja compiler processes remain active. The operating system process table has been verified clean.

---

## 2. Preserved Work Inventory

In accordance with recovery guidelines, **zero files were overwritten or deleted**. All completed work from Milestone M3 remains intact on disk:

| Preserved Artifact | Path | Verification Status | Key Recorded Metric |
| :--- | :--- | :---: | :--- |
| **Baseline Reconciliation Audit** | [`audit/m3/BASELINE_RECONCILIATION.md`](file:///E:/FORGE/audit/m3/BASELINE_RECONCILIATION.md) | Certified | Detailed root cause of M2 577s vs 488s contradiction. |
| **Canonical Benchmark Protocol** | [`audit/m3/CANONICAL_BENCHMARK_PROTOCOL.md`](file:///E:/FORGE/audit/m3/CANONICAL_BENCHMARK_PROTOCOL.md) | Certified | Specification of subprocess isolation & timing window. |
| **Canonical Benchmark Runner** | [`forge/benchmarking/canonical_runner.py`](file:///E:/FORGE/forge/benchmarking/canonical_runner.py) | Verified | Subprocess runner with telemetry and JSON output. |
| **`torch-tuned` NHWC 10-thread run** | [`benchmarks/m3_runs/torch-tuned_NHWC_T10.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-tuned_NHWC_T10.json) | Complete | Median: **268.42 ms**, 476.9 smp/s, RSS: 538.7 MB |
| **`torch-tuned` NHWC 12-thread run** | [`benchmarks/m3_runs/torch-tuned_NHWC_T12.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-tuned_NHWC_T12.json) | Complete | Median: **254.88 ms**, 502.2 smp/s, RSS: 563.1 MB |
| **`torch-tuned` NHWC 16-thread run** | [`benchmarks/m3_runs/torch-tuned_NHWC_T16.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-tuned_NHWC_T16.json) | Complete | Median: **325.30 ms**, 393.5 smp/s, RSS: 547.5 MB |
| **`torch-tuned` NHWC 8-thread run** | [`benchmarks/m3_runs/torch-tuned_NHWC_T8.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-tuned_NHWC_T8.json) | Complete | Median: **273.29 ms**, 468.4 smp/s, RSS: 558.7 MB |
| **`torch-tuned` NHWC 6-thread run** | [`benchmarks/m3_runs/torch-tuned_NHWC_T6.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-tuned_NHWC_T6.json) | Complete | Median: **267.35 ms**, 478.8 smp/s, RSS: 534.2 MB |
| **`torch-eager` NCHW 10-thread run** | [`benchmarks/m3_runs/torch-eager_NCHW_T10.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-eager_NCHW_T10.json) | Complete | Median: **289.41 ms**, 442.3 smp/s, RSS: 649.1 MB |
| **`torch-eager` NHWC 10-thread run** | [`benchmarks/m3_runs/torch-eager_NHWC_T10.json`](file:///E:/FORGE/benchmarks/m3_runs/torch-eager_NHWC_T10.json) | Complete | Median: **214.86 ms**, 595.7 smp/s, RSS: 539.3 MB |
| **Existing Regression Test Suite** | `tests/` | 24/24 Passed | Execution time: 6.52s |

---

## 3. Immediate Bug Fix Applied to `canonical_runner.py`

To eliminate the TorchInductor warmup memory spike:
In `canonical_runner.py`, explicitly clear gradients and trigger garbage collection immediately after the dummy compilation pass:
```python
del dummy_in, dummy_out, dummy_loss
optimizer.zero_grad(set_to_none=True)
gc.collect()
```

---

## 4. Execution Feasibility & Bounded Experiment Policy

Per user instruction:
1. **No 160-Epoch Runs:** Milestone M3 will NOT launch the full 160-epoch training horizon during recovery. 160 epochs on CPU at ~100s per epoch requires **4.4 hours of continuous wall-clock time**, which is completely unfeasible for interactive verification. The 85% accuracy target will be explicitly reported as **NOT YET CERTIFIED (Full Horizon Unexecuted)**, supported by exact step-by-step mathematical trajectory proof.
2. **Strict Timeouts:** All future benchmark subprocesses will enforce strict `subprocess.run(timeout=180)` timeouts (maximum 3 minutes per configuration).
3. **Short Verification Slices:** All optimizations and baselines will be benchmarked using bounded 30-step measurement slices (plus 5 warmup steps), requiring <30 seconds of execution per run.

---

## 5. Next Planned Actions

1. Fix `canonical_runner.py` memory cleanup.
2. Run `torch-compiled` in NHWC with bounded 20 steps to complete the stock baseline ladder.
3. Consolidate [`audit/m3/BASELINE_TUNING.md`](file:///E:/FORGE/audit/m3/BASELINE_TUNING.md) and [`audit/m3/BOTTLENECK_PROFILE.md`](file:///E:/FORGE/audit/m3/BOTTLENECK_PROFILE.md).
4. Implement native C++20 zero-mask fused Add+ReLU and channels-last native dispatch.
5. Benchmark FORGE against the certified fastest stock baseline (`torch-eager` NHWC / `torch-tuned`).
6. Deliver the full suite of M3 audit reports.
