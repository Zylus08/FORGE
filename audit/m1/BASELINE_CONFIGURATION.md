# FORGE M1 Baseline Configuration

## 1. Environment & Host Specification
- **Hardware Profile**: Refer to [HARDWARE_PROFILE.md](file:///E:/FORGE/audit/m1/HARDWARE_PROFILE.md)
  - **CPU**: Intel Core i7-12650H (Alder Lake, 6 P-cores + 4 E-cores = 10 physical cores, 16 logical threads)
  - **Memory**: 16 GB DDR4-3200 (Dual Channel)
  - **OS**: Windows 11 (25H2 Build 26200.9974)
  - **Python**: 3.12 (`reswork` conda environment)
  - **PyTorch**: 2.15.0+cpu (MKL oneAPI 2026.2, oneDNN v3.13.3, OpenMP, AVX2)

---

## 2. Baseline Ladder Definitions (B0 – B2)

### B0: `torch-eager` (Un-tuned Reference)
- **Description**: Standard PyTorch eager execution with default runtime thread pool configurations.
- **Thread Settings**: Default PyTorch thread settings (`torch.get_num_threads()` defaults to logical core count or PyTorch default; interop threads default).
- **Memory Format**: Default contiguous NCHW (`torch.contiguous_format`).
- **Data Loading**: Default PyTorch DataLoader (num_workers=0 or 2, pin_memory=False).
- **Target Purpose**: Forms the raw unoptimized baseline against which all engineering improvements and compiler benefits are measured.

### B1: `torch-tuned` (Host-Tuned Eager)
- **Description**: Hardware-aware eager mode optimization without graph compilation.
- **Thread Settings**:
  - `torch.set_num_threads(physical_cores)` (10 threads for 10 physical cores, avoiding hyperthread contention on P-cores and synchronization stalls).
  - `torch.set_num_interop_threads(1)` (prevents inter-op scheduling thrashing).
- **Memory Layout**: Memory format switched to `torch.channels_last` (NHWC layout) for both model weights and input activation tensors. This enables optimal vectorization with AVX2 and oneDNN conv kernels.
- **Data Loading**: Multi-process DataLoader (`num_workers=2`, `persistent_workers=True`) to overlap CPU batch preprocessing with model forward/backward passes.

### B2: `torch-compiled` (PyTorch 2.x TorchDynamo / TorchInductor)
- **Description**: Full graph capture and kernel generation using TorchDynamo with the TorchInductor CPU backend.
- **Configuration**:
  - `torch.compile(model, backend='inductor', mode='reduce-overhead')`
  - Threads tuned identically to B1 (`torch.set_num_threads(physical_cores)`, `torch.set_num_interop_threads(1)`).
- **Metrics Collected**:
  - Cold compilation time measured separately during warm-up.
  - Steady-state execution time per step and per epoch.

---

## 3. Workload Hyperparameters
- **Model**: ResNet-20 (`workloads/models/resnet20.py`, 272,474 parameters)
- **Dataset**: CIFAR-10 (50k train, 10k test, 32×32×3 RGB)
- **Batch Size**: 128
- **Loss**: `CrossEntropyLoss`
- **Optimizer**: SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
- **Scheduler**: MultiStepLR(milestones=[80, 120], gamma=0.1)
- **Timing Epochs**: 5 epochs (M1 protocol)
- **Warmup Steps**: 2 steps discarded from timing per epoch 1
- **Seeds**: PyTorch: 42, NumPy: 42, Random: 42
