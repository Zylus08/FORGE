# FORGE — PyTorch CPU Training Accelerator

> **Milestone M1** — Baseline infrastructure, hardware profiling, and benchmark protocol.

## Purpose

FORGE is a research accelerator that makes PyTorch CPU training faster without changing model code.  
It works by swapping in optimized backends (operator fusion, AVX-512/AMX kernels, oneDNN integration) transparently, while guaranteeing numerical equivalence with the standard `torch-eager` baseline.

## Milestone M1 Goals

| Goal | Description |
|------|-------------|
| Hardware audit | Collect CPU ISA, cache topology, RAM, and OS configuration |
| Baseline benchmarks | End-to-end throughput (samples/sec) for reference workloads on `torch-eager` |
| Numerical contract | Define tolerance bounds and verify bit-accuracy vs baseline |
| Scaffold | Complete project structure, CMake build, Python package |

## Directory Structure

```
FORGE/
├── forge/                  # Python package
│   ├── backends/           # Pluggable backend implementations
│   ├── profiling/          # CPU / memory profiling utilities
│   ├── benchmarking/       # Benchmark harness and runners
│   └── integration/        # PyTorch integration hooks
├── native/                 # C++20 kernel extensions
│   ├── include/            # Public headers
│   └── src/                # Kernel source files
├── workloads/              # Training workload definitions
│   ├── models/             # Model architectures
│   ├── datasets/           # Dataset loaders / synthetic generators
│   └── configs/            # Workload configuration YAML files
├── tests/                  # Test suite
│   ├── unit/               # Unit tests for individual components
│   ├── numerical/          # Numerical equivalence / accuracy tests
│   ├── integration/        # End-to-end integration tests
│   └── benchmark/          # Benchmark regression tests
├── benchmarks/             # Benchmark artifacts
│   ├── results/            # Stored JSON/CSV result files
│   └── scripts/            # Analysis and plotting scripts
├── audit/m1/               # M1 hardware & environment audit outputs
├── docs/                   # Architecture and protocol documentation
├── tools/                  # Developer utilities
├── pyproject.toml          # Python package metadata
└── CMakeLists.txt          # Native extension build
```

## Running Baselines

```bash
# Install in editable mode (dev extras)
pip install -e ".[dev]"

# Run the full baseline benchmark suite (torch-eager backend)
python -m forge.benchmarking.run_baselines --backend torch-eager

# Run a single workload
python -m forge.benchmarking.run_baselines --backend torch-eager --workload resnet18

# Save results to benchmarks/results/
python -m forge.benchmarking.run_baselines --backend torch-eager --out benchmarks/results/m1_baseline.json
```

## Running Tests

```bash
# Full test suite
pytest tests/

# Numerical equivalence only
pytest tests/numerical/ -v

# With coverage
pytest tests/ --cov=forge
```

## Building Native Extensions

Requirements: CMake ≥ 3.20, MSVC 2022 (or GCC 12+ on Linux), PyTorch C++ dev files.

```bash
# Configure
cmake -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# Build
cmake --build build --config Release

# The shared library lands in build/Release/forge_kernels.dll (Windows)
```

## License

MIT — see [LICENSE](LICENSE).
