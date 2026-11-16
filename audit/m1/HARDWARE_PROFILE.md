# FORGE M1 Hardware & Environment Profile

**Generated:** 2026-11-16  
**Host:** Windows 10 Home Single Language 25H2 (Build 26200.9974)

---

## 1. CPU

| Field                  | Value                                      |
|------------------------|--------------------------------------------|
| **Model**              | 12th Gen Intel(R) Core(TM) i7-12650H       |
| **Manufacturer**       | GenuineIntel                               |
| **Architecture**       | X86_64 (64-bit)                            |
| **Physical Cores**     | 10                                         |
| **Logical Processors** | 16                                         |
| **Max Clock Speed**    | 2300 MHz (base); 2688 MHz actual (cpuinfo) |
| **Socket**             | U3E1                                       |
| **Stepping**           | 3                                          |
| **Model Number**       | 154                                        |
| **Family**             | 6                                          |

### ISA Extensions (Runtime-Confirmed via py-cpuinfo v9.0.0)

```
3dnow, 3dnowprefetch, abm, acpi, adx, aes, apic,
avx, avx2,
bmi1, bmi2,
clflush, clflushopt, clwb, cmov, cx16, cx8,
de, dts, erms, est,
f16c, fma, fpu, fxsr,
gfni, ht, hypervisor, ia64, intel_pt, invpcid,
lahf_lm, mca, mce, mmx, monitor, movbe, msr, mtrr,
osxsave, pae, pat, pbe, pcid, pclmulqdq, pdcm, pge,
pni, popcnt, pse, pse36,
rdpid, rdrnd, rdseed, sep, serial, sha, smap, smep, ss,
sse, sse2, sse4_1, sse4_2, ssse3,
tm, tm2, tsc, tscdeadline, umip,
vaes, vme, vpclmulqdq,
x2apic, xsave, xtpr
```

**PyTorch CPU Capability (runtime):** `AVX2`

### Cache Hierarchy (Win32_CacheMemory — WMI Level encoding)

> WMI Level values: 3=L1, 4=L2, 5=L3

| WMI Level | Maps To | Size (KB) | Notes                         |
|-----------|---------|-----------|-------------------------------|
| 3         | L1      | 288       | P-core L1 (likely combined)   |
| 3         | L1      | 192       | E-core L1                     |
| 3         | L1      | 128       | P-core L1 (data or inst)      |
| 3         | L1      | 256       | Additional L1 segment         |
| 4         | L2      | 7680      | P-core L2 (7.5 MB)            |
| 4         | L2      | 2048      | E-core L2 (2 MB)              |
| 5         | L3      | 24576     | Shared L3 (24 MB) — ×2 slots |

> Note: WMI reports duplicate L3 entries (24576 KB each); physical L3 is 24 MB.

---

## 2. RAM

| Field           | Value                 |
|-----------------|-----------------------|
| **Total**       | 16 GB (2 × 8 GB)      |
| **Speed**       | 3200 MT/s             |
| **MemoryType**  | 0 (unreported by WMI) |
| **Manufacturer**| Nanya Technology      |
| **Slots**       | 2 DIMMs               |

> `wmic` is not available on this Windows build; speed confirmed via Win32_PhysicalMemory.

---

## 3. Operating System & Power

| Field            | Value                                              |
|------------------|----------------------------------------------------|
| **OS**           | Windows 10 Home Single Language                    |
| **Version**      | 25H2                                               |
| **Build**        | 26200.9974                                         |
| **NT Version**   | 10.0.26200.0                                       |
| **Power Plan**   | Balanced (GUID: 381b4222-f694-41f0-9685-ff5bb260df2e) |
| **CPU Temp**     | unknown (WMI MSAcpi_ThermalZoneTemperature: Access denied) |

---

## 4. Python & PyTorch Environment

> PyTorch is **not** installed in the base Anaconda environment. It is present in the **`reswork`** conda environment.

### Active Environment: `reswork`

| Field                   | Value                                                         |
|-------------------------|---------------------------------------------------------------|
| **Python Version**      | 3.14.2 (conda-forge, MSC v.1944 64-bit AMD64)                |
| **PyTorch Version**     | **2.15.0+cpu**                                                |
| **CUDA Available**      | False                                                         |
| **MKL Available**       | **True**                                                      |
| **MKL Version**         | Intel® oneAPI MKL 2026.2 (Build 20260915)                    |
| **oneDNN / MKLDNN**     | **True** — MKL-DNN v3.13.3                                   |
| **OpenMP**              | **True** — OpenMP 2019                                        |
| **LAPACK**              | Enabled (via MKL)                                             |
| **CPU Threads**         | 10                                                            |
| **CPU Interop Threads** | 10                                                            |
| **CPU Capability**      | **AVX2**                                                      |
| **BLAS Backend**        | MKL                                                           |

### Base Environment (anaconda3 base)

| Field               | Value              |
|---------------------|--------------------|
| **Python Version**  | 3.12.8             |
| **pip Version**     | 24.2               |
| **PyTorch**         | Not installed      |

### PyTorch Build Settings Summary

```
BLAS_INFO=mkl
BUILD_TYPE=Release
USE_MKL=ON
USE_MKLDNN=ON
USE_OPENMP=ON
USE_CUDA=0
USE_CUDNN=OFF
USE_GLOO=ON
USE_NCCL=OFF
PERF_WITH_AVX=1
PERF_WITH_AVX2=1
CXX_COMPILER=sccache-cl.exe (MSVC)
C++ Standard: C++20 (202002)
MSVC: 194234444
TORCH_VERSION=2.15.0
COMMIT_SHA=d0f2af8b66b20da4baa50644c0cd3f887a2eae2e
```

### Other Conda Environments

| Environment | Path                                             |
|-------------|--------------------------------------------------|
| base        | C:\Users\mishr\anaconda3                         |
| hiroshi     | C:\Users\mishr\anaconda3\envs\hiroshi            |
| myenv       | C:\Users\mishr\anaconda3\envs\myenv              |
| reswork     | C:\Users\mishr\anaconda3\envs\reswork (PyTorch!) |
| Op-Sindoor  | c:\Users\mishr\OneDrive\Desktop\Op-Sindoor\.conda|

---

## 5. Compiler / Toolchain

| Tool           | Version / Status                                                  |
|----------------|-------------------------------------------------------------------|
| **Python**     | 3.12.8 (base) / 3.14.2 (reswork)                                 |
| **pip**        | 24.2                                                              |
| **CMake**      | unknown (not found on PATH)                                       |
| **MSVC (cl)**  | unknown (not on PATH directly)                                    |
| **VS Install** | Visual Studio Build Tools 2022 v17.13.35919.96                   |
| **vswhere**    | 3.1.7+f39851e70f                                                  |
| **VC Tools**   | Present (Microsoft.VisualStudio.Component.VC.Tools.x86.x64)      |

> VS Build Tools 2022 is installed with MSVC C++ tools; `cl.exe` is accessible within a Developer Command Prompt but not on the global PATH.

---

## 6. Disk (C: Drive)

| Field    | Value                      |
|----------|----------------------------|
| **Used** | ~320.7 GB (320,730,566,656 bytes) |
| **Free** | ~241 MB (253,222,912 bytes) |

> ⚠️ **WARNING**: C: drive is critically low on free space (~241 MB). This may impact builds, temp files, and package installs.

---

## 7. Summary Table

| Fact                     | Value                              |
|--------------------------|------------------------------------|
| CPU Model                | Intel Core i7-12650H (Alder Lake-H)|
| Physical / Logical Cores | 10 / 16                            |
| Confirmed ISA Extensions | SSE2, SSE4.1, SSE4.2, AVX, **AVX2**, FMA, AES, SHA, BMI1, BMI2, F16C, VAES, GFNI, VPCLMULQDQ |
| PyTorch CPU Capability   | AVX2                               |
| RAM                      | 16 GB DDR4 @ 3200 MT/s             |
| L2 Cache                 | 7.5 MB (P) + 2 MB (E)             |
| L3 Cache                 | 24 MB shared                       |
| OS                       | Windows 10 25H2 (Build 26200.9974) |
| Power Plan               | Balanced                           |
| PyTorch Version          | 2.15.0+cpu (in `reswork` env)     |
| MKL                      | ✅ Present (oneAPI MKL 2026.2)     |
| oneDNN / MKL-DNN         | ✅ Present (v3.13.3)              |
| OpenMP                   | ✅ Present                         |
| CUDA                     | ❌ Not available                   |
| VS Build Tools           | 2022 v17.13 (MSVC 194x)           |
| C: Free Space            | ⚠️ ~241 MB (critically low)       |
