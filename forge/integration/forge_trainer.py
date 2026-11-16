"""FORGE Integration: Model wrapper and trainer for native accelerated execution."""

from typing import Dict, List, Optional, Tuple
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.backends.forge_backend import (
    FusedReLU,
    FusedReLUBias,
    ForgePatchContext,
    forge_replace_ops,
    get_native_ops,
)


class ForgeAcceleratedModel(nn.Module):
    """Wrapper that ensures all ReLU operations use native C++ fused kernels."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # Replace any nn.ReLU module instances
        self.model = forge_replace_ops(model)
        # Ensure native ops are loaded
        get_native_ops()

    def forward(self, *args, **kwargs):
        # Intercept functional F.relu calls in the model forward pass
        with ForgePatchContext():
            return self.model(*args, **kwargs)


def apply_forge_backend(model: nn.Module) -> nn.Module:
    """Applies the FORGE native acceleration backend to a PyTorch model."""
    # Ensure native extension is compiled and loadable
    get_native_ops()
    return ForgeAcceleratedModel(model)


class ForgeTrainer:
    """Trainer wrapping model execution with the FORGE native backend."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device = torch.device("cpu"),
    ):
        self.raw_model = model
        self.model = apply_forge_backend(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        warmup_steps: int = 2,
    ) -> Dict:
        """Run one training epoch with step timing collection."""
        self.model.train()

        step_times: List[float] = []
        data_times: List[float] = []
        fwd_times: List[float] = []
        bwd_times: List[float] = []
        opt_times: List[float] = []

        correct = 0
        total = 0
        step_idx = 0

        t_epoch_start = time.perf_counter()
        t_data_start = time.perf_counter()

        for inputs, targets in loader:
            t_data_end = time.perf_counter()
            data_load = t_data_end - t_data_start

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward
            t_fwd_start = time.perf_counter()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            t_fwd_end = time.perf_counter()

            # Backward
            self.optimizer.zero_grad()
            t_bwd_start = time.perf_counter()
            loss.backward()
            t_bwd_end = time.perf_counter()

            # Optimizer
            t_opt_start = time.perf_counter()
            self.optimizer.step()
            t_opt_end = time.perf_counter()

            step_total = t_opt_end - t_data_start

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            if step_idx >= warmup_steps:
                step_times.append(step_total)
                data_times.append(data_load)
                fwd_times.append(t_fwd_end - t_fwd_start)
                bwd_times.append(t_bwd_end - t_bwd_start)
                opt_times.append(t_opt_end - t_opt_start)

            step_idx += 1
            t_data_start = time.perf_counter()

        epoch_time = time.perf_counter() - t_epoch_start
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        samples_per_sec = (len(loader.dataset) / epoch_time) if epoch_time > 0 else 0.0

        return {
            "epoch_time": epoch_time,
            "accuracy": accuracy,
            "samples_per_sec": samples_per_sec,
            "step_times": step_times,
            "data_times": data_times,
            "fwd_times": fwd_times,
            "bwd_times": bwd_times,
            "opt_times": opt_times,
        }
