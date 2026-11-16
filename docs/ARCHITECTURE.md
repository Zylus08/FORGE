# FORGE Backend Architecture

## Overview

FORGE uses a **pluggable backend** system. Every backend implements a common interface so the benchmark harness, profiler, and accuracy checker can treat them identically.

## Backend Interface

Each backend lives in `forge/backends/<name>.py` and must expose:

```python
class ForgeBackend:
    name: str                    # e.g. "torch-eager", "forge-avx512"
    def prepare(self, model, config) -> PreparedModel: ...
    def train_step(self, prepared, batch) -> StepResult: ...
    def teardown(self, prepared): ...
```

`PreparedModel` wraps the original `nn.Module` plus any backend-specific state (compiled graph, kernel handles, thread-pool references).  
`StepResult` contains: `loss`, `throughput_samples_per_sec`, `wall_time_ms`, `cpu_time_ms`.

## Backend Registry

Backends self-register on import:

```python
# forge/backends/_registry.py
_REGISTRY: dict[str, type[ForgeBackend]] = {}

def register(cls):
    _REGISTRY[cls.name] = cls
    return cls
```

Calling `forge.get_backend("forge-avx512")` looks up the registry and instantiates the backend.

## Built-in Backends (M1 scope)

| Backend name | Description |
|---|---|
| `torch-eager` | Unmodified PyTorch eager mode — the reference baseline |
| `torch-compile` | `torch.compile()` with default settings |
| `forge-onednn` | Explicitly routes ops through oneDNN via `torch.backends.mkldnn` |

## FORGE Mode

"FORGE mode" is activated by calling `forge.optimize(model)`. It:

1. Inspects the model graph for fusable patterns (e.g., Linear → ReLU, BN → ReLU).
2. Replaces matched subgraphs with FORGE fused kernels from `forge_kernels.dll`.
3. Installs gradient hooks so backward also uses fused kernels where possible.
4. Returns the patched module — the calling code is otherwise unchanged.

FORGE mode is opt-in and always verifiable against `torch-eager` via `forge.verify(model, inputs)`.

## Threading Model

FORGE respects `torch.set_num_threads()`. The native kernels use OpenMP for intra-op parallelism and defer to PyTorch's inter-op thread pool for async dispatch.

## Adding a New Backend

1. Create `forge/backends/my_backend.py`.
2. Subclass `ForgeBackend`, set `name = "my-backend"`.
3. Decorate with `@forge.backends.register`.
4. Add an entry to `workloads/configs/backends.yaml`.
5. Write at least one numerical test in `tests/numerical/test_my_backend.py`.
