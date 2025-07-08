"""
Tests for audioloop.utils.data_utils module.

This module contains function-based tests for the core data utilities
used throughout the AudioLoop active learning framework.
"""

import pytest
import torch

from audioloop.utils.data_utils import entropy, get_device, simple_collate_fn


# Test fixtures
@pytest.fixture
def sample_mono_batch():
    """Create sample batch with mono spectrograms."""
    return [
        {
            "data": torch.randn(1, 64, 100),
            "label": 1,
            "filename": "test1.pt",
            "filepath": "path/test1.pt",
        },
        {
            "data": torch.randn(1, 64, 100),
            "label": 0,
            "filename": "test2.pt",
            "filepath": "path/test2.pt",
        },
    ]


# Entropy function tests
def test_entropy_uniform_distribution():
    """Test entropy calculation for uniform binary distribution."""
    probs = torch.tensor([[0.5, 0.5]])
    result = entropy(probs)
    # For uniform binary: entropy = -2*(0.5*log(0.5)) ≈ 0.693
    assert torch.allclose(result, torch.tensor([0.693]), atol=1e-3)


def test_entropy_certain_distribution():
    """Test entropy calculation for certain distribution."""
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    result = entropy(probs)
    # Certain distribution should have zero entropy
    assert torch.allclose(result, torch.tensor([0.0, 0.0]), atol=1e-6)


def test_entropy_handles_zero_probabilities():
    """Test that entropy handles zero probabilities without NaN."""
    probs = torch.tensor([[1.0, 0.0]])
    result = entropy(probs)

    assert not torch.any(torch.isnan(result))
    assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)


def test_entropy_batch_processing():
    """Test entropy calculation preserves batch dimension."""
    batch_size = 5
    num_classes = 3
    probs = torch.rand(batch_size, num_classes)
    probs = probs / probs.sum(dim=1, keepdim=True)  # Normalize to probabilities

    result = entropy(probs)

    assert result.shape == (batch_size,)
    assert torch.all(result >= 0)  # Entropy should be non-negative


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([[0.5, 0.5]], [0.693]),
        ([[1.0, 0.0]], [0.0]),
        ([[0.8, 0.2]], [0.5]),
        ([[0.25, 0.25, 0.25, 0.25]], [1.386]),  # 4-class uniform
    ],
)
def test_entropy_parametrized(probs, expected):
    """Test entropy for various probability distributions."""
    probs_tensor = torch.tensor(probs)
    result = entropy(probs_tensor)
    expected_tensor = torch.tensor(expected)
    assert torch.allclose(result, expected_tensor, atol=1e-1)


# Device selection tests
def test_get_device_returns_valid_device():
    """Test that get_device returns a valid torch device."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda", "mps"]


def test_get_device_prefers_cuda(mocker):
    """Test that CUDA is preferred when available."""
    mocker.patch("torch.cuda.is_available", return_value=True)
    device = get_device()
    assert device.type == "cuda"


def test_get_device_falls_back_to_mps(mocker):
    """Test that MPS is used when CUDA unavailable."""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    device = get_device()
    assert device.type == "mps"


def test_get_device_falls_back_to_cpu(mocker):
    """Test that CPU is used as final fallback."""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    device = get_device()
    assert device.type == "cpu"


# Simple collate function tests
def test_simple_collate_basic_functionality(sample_mono_batch):
    """Test basic collation of mono spectrograms."""
    result = simple_collate_fn(sample_mono_batch)

    assert result["data"].shape == (2, 1, 64, 100)
    assert torch.equal(result["label"], torch.tensor([1, 0]))
    assert result["filename"] == ["test1.pt", "test2.pt"]
    assert result["filepath"] == ["path/test1.pt", "path/test2.pt"]


def test_simple_collate_single_item():
    """Test collation with single item batch."""
    batch = [
        {
            "data": torch.randn(1, 64, 100),
            "label": 1,
            "filename": "single.pt",
            "filepath": "path/single.pt",
        }
    ]

    result = simple_collate_fn(batch)

    assert result["data"].shape == (1, 1, 64, 100)
    assert result["label"].shape == (1,)
    assert len(result["filename"]) == 1
