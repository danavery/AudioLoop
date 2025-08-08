"""
Tests for audioloop.utils.data_utils module.

This module contains function-based tests for the core data utilities
used throughout the AudioLoop active learning framework.
"""

import pytest
import torch

from audioloop.utils.data_utils import (
    entropy,
    get_device,
    simple_collate_fn,
    variable_length_collate_fn,
)


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


class TestVariableLengthCollate:
    """Test variable_length_collate_fn functionality."""

    @pytest.fixture
    def variable_length_batch(self):
        """Sample batch with different length spectrograms."""
        return [
            {
                "data": torch.randn(1, 64, 100),  # Short spectrogram
                "label": 1,
                "filename": "short.pt",
                "filepath": "path/short.pt",
            },
            {
                "data": torch.randn(1, 64, 150),  # Longer spectrogram
                "label": 0,
                "filename": "long.pt",
                "filepath": "path/long.pt",
            },
        ]

    @pytest.fixture
    def same_length_batch(self):
        """Sample batch where all spectrograms are same length."""
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

    def test_variable_length_collate_basic(self, variable_length_batch):
        """Test collation of variable length spectrograms."""
        result = variable_length_collate_fn(variable_length_batch)

        # Should pad to max length (150)
        assert result["data"].shape == (2, 1, 64, 150)
        assert torch.equal(result["label"], torch.tensor([1, 0]))
        assert result["filename"] == ["short.pt", "long.pt"]
        assert result["filepath"] == ["path/short.pt", "path/long.pt"]

        # Check that shorter spectrogram was padded with zeros
        short_spec = result["data"][0]  # First item (originally 100 frames)
        # Last 50 frames should be zero (padding)
        assert torch.all(short_spec[..., 100:] == 0)
        # First 100 frames should not be all zero (original data)
        assert not torch.all(short_spec[..., :100] == 0)

    def test_variable_length_collate_same_length(self, same_length_batch):
        """Test that variable collate works with same-length spectrograms."""
        result = variable_length_collate_fn(same_length_batch)

        # Should work exactly like simple_collate_fn for same lengths
        assert result["data"].shape == (2, 1, 64, 100)
        assert torch.equal(result["label"], torch.tensor([1, 0]))
        assert result["filename"] == ["test1.pt", "test2.pt"]
        assert result["filepath"] == ["path/test1.pt", "path/test2.pt"]

    def test_variable_length_collate_single_item(self):
        """Test variable collate with single item batch."""
        batch = [
            {
                "data": torch.randn(1, 64, 75),
                "label": 1,
                "filename": "single.pt",
                "filepath": "path/single.pt",
            }
        ]

        result = variable_length_collate_fn(batch)

        assert result["data"].shape == (1, 1, 64, 75)  # No padding needed
        assert result["label"].shape == (1,)
        assert len(result["filename"]) == 1

    def test_variable_length_collate_three_different_lengths(self):
        """Test with three spectrograms of different lengths."""
        batch = [
            {
                "data": torch.randn(1, 64, 50),  # Short
                "label": 1,
                "filename": "short.pt",
                "filepath": "path/short.pt",
            },
            {
                "data": torch.randn(1, 64, 200),  # Long
                "label": 0,
                "filename": "long.pt",
                "filepath": "path/long.pt",
            },
            {
                "data": torch.randn(1, 64, 125),  # Medium
                "label": 1,
                "filename": "medium.pt",
                "filepath": "path/medium.pt",
            },
        ]

        result = variable_length_collate_fn(batch)

        # Should pad all to max length (200)
        assert result["data"].shape == (3, 1, 64, 200)
        assert torch.equal(result["label"], torch.tensor([1, 0, 1]))

        # Check padding
        short_spec = result["data"][0]  # Originally 50 frames
        medium_spec = result["data"][2]  # Originally 125 frames

        # Short should have padding from 50 to 200
        assert torch.all(short_spec[..., 50:] == 0)
        # Medium should have padding from 125 to 200
        assert torch.all(medium_spec[..., 125:] == 0)
