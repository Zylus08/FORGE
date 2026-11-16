# FORGE Milestone M2: Known Limitations & Technical Risks

**Date:** 2026-11-17  
**Auditor:** Principal Systems Auditor  
**Artifact Location:** [`audit/m2/M2_LIMITATIONS.md`](file:///E:/FORGE/audit/m2/M2_LIMITATIONS.md)

---

## 1. Workload Scope & Architecture Limitations

1. **Fixed Architecture Specialization:**
   The M2 native kernels (`fused_relu`, `fused_add_relu`, `fused_relu_bias`) are specialized for residual convolutional topologies (specifically ResNet-20). While generalizable to other ResNet variants (ResNet-18, ResNet-34, ResNet-50), transformer architectures (e.g. self-attention, layer normalization, GELU) require distinct kernel fusion pipelines.
2. **Precision Bound:**
   All M2 implementations strictly adhere to FP32 single-precision floating point. Mixed-precision (BF16 / FP16) and quantization (INT8) training optimizations were intentionally out of scope for M2 to preserve the exact scientific contract.

---

## 2. Hardware & Platform Constraints

1. **Host Storage Constraints (Drive C: vs Drive E:):**
   Drive C: on the host machine has limited free space (~1.08 GB free). All build directories, CMake builds, torchvision dataset downloads, and profiler traces must remain pinned to Drive E: (`E:\FORGE`).
2. **Windows Toolchain Environment Requirements:**
   TorchInductor C++ code generation on Windows 11 strictly requires MSVC `cl.exe`, Microsoft OpenMP runtime (`/openmp`), and Windows Kits 10 UCRT/shared headers. While we integrated automatic path resolution into `run_baselines.py`, users running custom standalone scripts must activate `vcvarsall.bat x64` or ensure SDK paths are set in `INCLUDE` and `LIB`.
3. **CPU Heterogeneity (Alder Lake P-cores vs. E-cores):**
   The Intel Core i7-12650H features 6 Performance cores (12 hyperthreads) and 4 Efficient cores. Thread counts between 4 and 8 on standard NCHW convolutions suffer from thread scheduling lock contention. The optimal thread allocation is 10 threads, matching physical core count.

---

## 3. Mathematical & Convergence Boundaries

1. **Five-Epoch Horizon vs. 160-Epoch Full Target:**
   Milestone M2 benchmarks were executed over 5-epoch horizons to measure steady-state training throughput, per-step timing distributions, and initial trajectory stability. While test accuracy tracked identically between PyTorch eager and FORGE at epoch 1 (47.09%) and followed consistent learning rate dynamics, full 160-epoch convergence to the target $\ge 85.0\%$ accuracy criterion remains to be verified upon full-horizon execution.
2. **Amdahl Speedup Upper Bound:**
   Because convolution operations constitute ~65.7% of the total execution budget, acceleration of activation and residual paths alone is mathematically bounded at $\approx 1.08\times$ against compiled PyTorch. The 2.5× objective requires convolution-level fusion in Milestone M3.
