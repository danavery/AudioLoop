"""
Tests for training core functionality.

This module tests class weighting and other training-related features.
"""

import torch

from audioloop.config import AudioLoopConfig


class TestClassWeightingConfig:
    """Test class weighting configuration."""

    def test_class_weighting_disabled_by_default(self):
        """Test that class weighting is disabled by default."""
        config = AudioLoopConfig()
        assert config.class_weighting is None

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
        try:
            config = AudioLoopConfig(class_weighting="invalid")
            config._validate_training_params()
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "class_weighting must be 'adaptive'" in str(e)

    def test_class_weighting_validation_rejects_out_of_range_float(self):
        """Test that out-of-range float values are rejected."""
        try:
            config = AudioLoopConfig(class_weighting=1.5)
            config._validate_training_params()
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "between 0.0 and 1.0" in str(e)


class TestClassWeightingCalculation:
    """Test class weighting calculation logic."""

    def test_class_weight_calculation_balanced_dataset(self):
        """Test weight calculation for balanced dataset."""
        # Simulate balanced dataset (50 samples each class)
        labels = [0] * 50 + [1] * 50
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())

        # For balanced dataset, weights should be equal
        expected_weights = torch.tensor([1.0, 1.0])
        assert torch.allclose(class_weights, expected_weights, atol=1e-6)

    def test_class_weight_calculation_imbalanced_dataset(self):
        """Test weight calculation for imbalanced dataset."""
        # Simulate imbalanced dataset (10 positive, 90 negative)
        labels = [0] * 90 + [1] * 10
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())

        # Expected: total_samples / (num_classes * class_counts)
        # Class 0 (90 samples): 100 / (2 * 90) = 0.556
        # Class 1 (10 samples): 100 / (2 * 10) = 5.0
        expected_weights = torch.tensor([100.0 / (2 * 90), 100.0 / (2 * 10)])
        assert torch.allclose(class_weights, expected_weights, atol=1e-6)

        # Minority class should have much higher weight
        assert class_weights[1] > class_weights[0]
        assert abs(class_weights[1] / class_weights[0] - 9.0) < 0.1  # Should be ~9x higher

    def test_class_weight_calculation_extreme_imbalance(self):
        """Test weight calculation with extreme class imbalance."""
        # Simulate extremely imbalanced dataset (1 positive, 999 negative)
        labels = [0] * 999 + [1] * 1
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())

        # Minority class should get extremely high weight
        assert class_weights[1] > class_weights[0]

        # Expected ratio: Class 0 weight = 1000/(2*999), Class 1 weight = 1000/(2*1)
        # Ratio = (1000/(2*1)) / (1000/(2*999)) = 999/1 = 999
        expected_ratio = 999.0
        actual_ratio = class_weights[1] / class_weights[0]
        assert abs(actual_ratio - expected_ratio) < 1.0

    def test_class_weight_calculation_single_class(self):
        """Test weight calculation with single class dataset."""
        # Single class (all positive)
        labels = [1] * 100
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)

        # Pad class_counts to ensure we have 2 classes for binary classification
        if len(class_counts) == 1:
            class_counts = torch.cat([torch.tensor([0]), class_counts])

        class_weights = total_samples / (len(class_counts) * class_counts.float())

        # Class 0 should have inf weight (no samples), class 1 should have finite weight
        assert torch.isinf(class_weights[0])
        assert torch.isfinite(class_weights[1])

    def test_class_weight_integration_with_crossentropy(self):
        """Test that class weights can be passed to CrossEntropyLoss."""
        labels = [0] * 90 + [1] * 10
        class_counts = torch.bincount(torch.tensor(labels))
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())

        # Should be able to create weighted loss function
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Test that it works with dummy data
        dummy_logits = torch.randn(10, 2)  # batch_size=10, num_classes=2
        dummy_labels = torch.randint(0, 2, (10,))

        loss = criterion(dummy_logits, dummy_labels)
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # Loss should be non-negative
