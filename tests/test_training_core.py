"""
Tests for training core functionality.

This module tests class weighting and other training-related features.
"""

from typing import cast

import pytest
import torch

from audioloop.config import AudioLoopConfig
from audioloop.training_core import setup_loss_criterion
from audioloop.utils.spectrogram_dataset import SpectrogramDataset


class _FakeTrainDataset:
    """Minimal stand-in for SpectrogramDataset.

    setup_loss_criterion only reads `.samples` (for the per-sample labels) and `len(...)`
    (for the total), so a list of {"label": ...} dicts is a faithful substitute that keeps
    the test off the disk/audio path.
    """

    def __init__(self, labels):
        self.samples = [{"label": label} for label in labels]

    def __len__(self):
        return len(self.samples)


_CPU = torch.device("cpu")


def _criterion_for(config: AudioLoopConfig, labels: list[int]) -> torch.nn.CrossEntropyLoss:
    """Build the loss criterion for a config + label distribution via the real factory.

    Casts the lightweight fake to SpectrogramDataset (setup_loss_criterion only touches the
    duck-typed surface) and narrows the nn.Module return to CrossEntropyLoss so callers can
    read `.weight` directly — and so the test pins the factory's return contract.
    """
    dataset = cast(SpectrogramDataset, _FakeTrainDataset(labels))
    criterion = setup_loss_criterion(config, dataset, _CPU)
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    return criterion


class TestClassWeightingConfig:
    """Test class weighting configuration."""

    def test_class_weighting_default(self):
        """Test that class weighting defaults to 0.70."""
        config = AudioLoopConfig()
        assert config.class_weighting == 0.70

    def test_class_weighting_adaptive_mode(self):
        """Test that adaptive class weighting can be enabled."""
        config = AudioLoopConfig(class_weighting="adaptive")
        assert config.class_weighting == "adaptive"

    def test_class_weighting_fixed_mode(self):
        """Test that fixed class weighting can be configured."""
        config = AudioLoopConfig(class_weighting=0.25)
        assert config.class_weighting == 0.25

    def test_class_weighting_validation_rejects_invalid_string(self):
        """Test that invalid string values are rejected."""
        with pytest.raises(ValueError, match="class_weighting must be 'adaptive'"):
            AudioLoopConfig(class_weighting="invalid")

    def test_class_weighting_validation_rejects_out_of_range_float(self):
        """Test that out-of-range float values are rejected."""
        with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
            AudioLoopConfig(class_weighting=1.5)


class TestClassWeightingCalculation:
    """Test the loss criterion built by setup_loss_criterion (the real production path).

    These exercise setup_loss_criterion end-to-end and assert on the weight tensor it
    attaches to the returned CrossEntropyLoss, rather than re-deriving the weighting formula
    in the test (which would pass even if training_core's formula were wrong).
    """

    def test_adaptive_balanced_dataset_gives_equal_weights(self):
        """Adaptive weighting on a 50/50 split yields equal class weights."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 50 + [1] * 50)

        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, torch.tensor([1.0, 1.0]), atol=1e-6)

    def test_adaptive_imbalanced_dataset_upweights_minority(self):
        """Adaptive weighting on a 90/10 split upweights the minority class ~9x."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 90 + [1] * 10)

        # total / (num_classes * count): [100/(2*90), 100/(2*10)] = [0.5556, 5.0]
        expected = torch.tensor([100.0 / (2 * 90), 100.0 / (2 * 10)])
        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, expected, atol=1e-6)
        assert criterion.weight[1] > criterion.weight[0]
        assert abs(criterion.weight[1] / criterion.weight[0] - 9.0) < 0.1

    def test_adaptive_extreme_imbalance(self):
        """Adaptive weighting on a 999/1 split produces a ~999x weight ratio."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 999 + [1] * 1)

        assert criterion.weight is not None
        assert criterion.weight[1] > criterion.weight[0]
        assert abs(criterion.weight[1] / criterion.weight[0] - 999.0) < 1.0

    def test_fixed_float_weighting_targets_positive_ratio(self):
        """A fixed float weighting sets pos weight = (1 - w) / w with neg weight 1.0."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting=0.25), [0] * 90 + [1] * 10)

        # weight_neg=1.0, weight_pos=(1-0.25)/0.25=3.0; independent of the actual counts.
        assert criterion.weight is not None
        assert torch.allclose(criterion.weight, torch.tensor([1.0, 3.0]), atol=1e-6)

    def test_none_weighting_produces_unweighted_loss(self):
        """class_weighting=None yields an unweighted CrossEntropyLoss (weight is None)."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting=None), [0] * 90 + [1] * 10)

        assert criterion.weight is None

    def test_weighted_criterion_is_usable(self):
        """The criterion setup_loss_criterion returns produces a finite, non-negative loss."""
        criterion = _criterion_for(AudioLoopConfig(class_weighting="adaptive"), [0] * 90 + [1] * 10)

        dummy_logits = torch.randn(10, 2)  # batch_size=10, num_classes=2
        dummy_labels = torch.randint(0, 2, (10,))
        loss = criterion(dummy_logits, dummy_labels)

        assert torch.isfinite(loss)
        assert loss.item() >= 0
