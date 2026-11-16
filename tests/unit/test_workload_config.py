"""Unit tests for workload configuration and model architecture reproducibility."""
import yaml
import pytest
import torch
import torchvision.transforms as transforms
from pathlib import Path

from workloads.models.resnet20 import resnet20
from forge.benchmarking.run_baselines import set_seeds

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "workloads" / "configs" / "primary.yaml"


def test_workload_config_loads_correctly():
    """Verify that workloads/configs/primary.yaml exists and contains required fields."""
    assert CONFIG_PATH.exists(), f"Configuration file {CONFIG_PATH} not found"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["workload_id"] == "forge-m1-primary"
    assert cfg["model"] == "resnet20"
    assert cfg["dataset"] == "cifar10"
    assert cfg["num_classes"] == 10
    assert cfg["batch_size"] == 128
    assert "optimizer" in cfg
    assert cfg["optimizer"]["name"] == "SGD"
    assert cfg["optimizer"]["lr"] == 0.01
    assert cfg["optimizer"]["momentum"] == 0.9
    assert cfg["optimizer"]["weight_decay"] == 0.0001
    assert "scheduler" in cfg
    assert cfg["scheduler"]["name"] == "MultiStepLR"
    assert cfg["scheduler"]["milestones"] == [80, 120]
    assert cfg["seed"] == 42
    assert cfg["m1_timing_epochs"] == 5
    assert cfg["accuracy_target_epoch160"] == 85.0


def test_model_parameter_count():
    """Verify that ResNet-20 has exactly 269,722 parameters."""
    model = resnet20()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 269722, f"Expected 269,722 parameters, got {total_params}"

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 269722, f"Expected all 269,722 parameters to be trainable, got {trainable_params}"


def test_seed_setting_reproducibility():
    """Verify that setting seeds produces deterministic torch and numpy outputs."""
    set_seeds(42)
    t1 = torch.randn(10, 10)
    set_seeds(42)
    t2 = torch.randn(10, 10)
    assert torch.equal(t1, t2), "torch.manual_seed did not produce deterministic output"

    set_seeds(1234)
    t3 = torch.randn(10, 10)
    assert not torch.equal(t1, t3), "Different seeds produced identical tensors"


def test_dataset_transforms_deterministic():
    """Verify that data normalization transform is deterministic on given tensor."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    raw_tensor = torch.ones(3, 32, 32) * 0.5
    out1 = normalize(raw_tensor.clone())
    out2 = normalize(raw_tensor.clone())
    assert torch.equal(out1, out2), "Normalize transform was non-deterministic"
