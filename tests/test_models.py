"""Tests for the AudioLoop model system."""

import tempfile
import torch
import pytest
from pathlib import Path

from audioloop.models import AudioLoopModel, SoundCNN, SimpleCNN


class TestAudioLoopModel:
    """Test the AudioLoopModel ABC."""

    def test_is_abstract(self):
        """Test that AudioLoopModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AudioLoopModel()  # pyright: ignore[reportAbstractUsage]

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteModel(AudioLoopModel):
            pass

        with pytest.raises(TypeError):
            IncompleteModel()  # pyright: ignore[reportAbstractUsage]

    def test_complete_subclass_can_be_instantiated(self):
        """Test that complete subclasses can be instantiated."""

        class CompleteModel(torch.nn.Module, AudioLoopModel):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)

            def save_model(self, path: str) -> None:
                torch.save({"test": "data"}, path)

            @classmethod
            def load_model(cls, path: str, device: torch.device) -> "CompleteModel":
                return cls()

            def get_model_info(self) -> dict:
                return {"type": "test"}

        # Should not raise an exception
        model = CompleteModel()
        assert isinstance(model, AudioLoopModel)
        assert isinstance(model, torch.nn.Module)


class TestSoundCNN:
    """Test the SoundCNN model implementation."""

    def test_basic_initialization(self):
        """Test basic SoundCNN initialization."""
        model = SoundCNN(num_classes=2)

        assert model.num_classes == 2
        assert model.kernel_size == (3, 3)
        assert model.batchnorm_threshold == 100
        assert model.use_batchnorm is True  # Default when no dataset_size
        assert isinstance(model, AudioLoopModel)
        assert isinstance(model, torch.nn.Module)

    def test_batchnorm_decision_large_dataset(self):
        """Test BatchNorm decision for large datasets."""
        model = SoundCNN(num_classes=2, dataset_size=150)

        assert model.use_batchnorm is True
        assert hasattr(model, 'bn1')
        assert hasattr(model, 'bn2')

    def test_batchnorm_decision_small_dataset(self):
        """Test BatchNorm decision for small datasets."""
        model = SoundCNN(num_classes=2, dataset_size=50)

        assert model.use_batchnorm is False
        assert not hasattr(model, 'bn1')
        assert not hasattr(model, 'bn2')
        assert hasattr(model, 'dropout')  # Should have dropout instead

    def test_batchnorm_decision_at_threshold(self):
        """Test BatchNorm decision exactly at threshold."""
        model = SoundCNN(num_classes=2, dataset_size=100)

        assert model.use_batchnorm is True

        model_below = SoundCNN(num_classes=2, dataset_size=99)
        assert model_below.use_batchnorm is False

    def test_custom_batchnorm_threshold(self):
        """Test custom BatchNorm threshold."""
        model = SoundCNN(num_classes=2, dataset_size=50, batchnorm_threshold=40)

        assert model.use_batchnorm is True
        assert model.batchnorm_threshold == 40

    def test_custom_threshold_extreme_values(self):
        """Test extreme threshold values."""
        # Always use BatchNorm
        model_always = SoundCNN(num_classes=2, dataset_size=1, batchnorm_threshold=0)
        assert model_always.use_batchnorm is True

        # Never use BatchNorm
        model_never = SoundCNN(num_classes=2, dataset_size=1000, batchnorm_threshold=float('inf'))
        assert model_never.use_batchnorm is False

    def test_no_dataset_size_defaults_to_batchnorm(self):
        """Test that missing dataset_size defaults to using BatchNorm."""
        model = SoundCNN(num_classes=2)

        assert model.use_batchnorm is True

    def test_custom_parameters(self):
        """Test custom kernel size and other parameters."""
        model = SoundCNN(
            num_classes=5,
            kernel_size=(5, 5),
            dataset_size=200,
            batchnorm_threshold=150
        )

        assert model.num_classes == 5
        assert model.kernel_size == (5, 5)
        assert model.batchnorm_threshold == 150
        assert model.use_batchnorm is True

    def test_forward_pass_with_batchnorm(self):
        """Test forward pass with BatchNorm enabled."""
        model = SoundCNN(num_classes=2, dataset_size=150)

        # Create a sample input (batch=1, channels=1, height=128, width=128)
        x = torch.randn(1, 1, 128, 128)
        output = model(x)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_forward_pass_without_batchnorm(self):
        """Test forward pass without BatchNorm."""
        model = SoundCNN(num_classes=2, dataset_size=50)

        # Create a sample input
        x = torch.randn(1, 1, 128, 128)
        output = model(x)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_model_info(self):
        """Test get_model_info method."""
        model = SoundCNN(num_classes=3, dataset_size=75, batchnorm_threshold=80)

        info = model.get_model_info()

        assert info["model_type"] == "SoundCNN"
        assert info["num_classes"] == 3
        assert info["kernel_size"] == (3, 3)
        assert info["use_batchnorm"] is False  # 75 < 80
        assert info["batchnorm_threshold"] == 80
        assert "num_parameters" in info
        assert info["num_parameters"] > 0

    def test_parameter_count_difference(self):
        """Test that BatchNorm affects parameter count."""
        model_with_bn = SoundCNN(num_classes=2, dataset_size=150)
        model_without_bn = SoundCNN(num_classes=2, dataset_size=50)

        params_with_bn = model_with_bn.get_model_info()["num_parameters"]
        params_without_bn = model_without_bn.get_model_info()["num_parameters"]

        assert params_with_bn > params_without_bn

    def test_save_model(self):
        """Test model saving functionality."""
        model = SoundCNN(num_classes=2, dataset_size=50, batchnorm_threshold=75)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            model.save_model(temp_path)

            # Verify file exists
            assert Path(temp_path).exists()

            # Verify file contains expected data
            checkpoint = torch.load(temp_path, map_location='cpu')
            assert checkpoint["model_type"] == "SoundCNN"
            assert checkpoint["num_classes"] == 2
            assert checkpoint["kernel_size"] == (3, 3)
            assert checkpoint["use_batchnorm"] is False
            assert checkpoint["batchnorm_threshold"] == 75
            assert "model_state_dict" in checkpoint

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_model(self):
        """Test model loading functionality."""
        original_model = SoundCNN(num_classes=3, dataset_size=25, batchnorm_threshold=50)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            # Save the model
            original_model.save_model(temp_path)

            # Load it back
            device = torch.device('cpu')
            loaded_model = SoundCNN.load_model(temp_path, device)

            # Verify loaded model has same properties
            assert loaded_model.num_classes == original_model.num_classes
            assert loaded_model.kernel_size == original_model.kernel_size
            assert loaded_model.use_batchnorm == original_model.use_batchnorm
            assert loaded_model.batchnorm_threshold == original_model.batchnorm_threshold

            # Verify model info matches
            original_info = original_model.get_model_info()
            loaded_info = loaded_model.get_model_info()
            assert original_info == loaded_info

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_model_preserves_batchnorm_decision(self):
        """Test that loading preserves the original BatchNorm decision."""
        # Create a model where dataset_size < threshold but we want to load
        # the exact same architecture that was saved
        model = SoundCNN(num_classes=2, dataset_size=50, batchnorm_threshold=75)
        assert model.use_batchnorm is False

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            model.save_model(temp_path)

            # Load model - should preserve the original BatchNorm decision
            device = torch.device('cpu')
            loaded_model = SoundCNN.load_model(temp_path, device)

            # Should preserve the original decision, not recalculate
            assert loaded_model.use_batchnorm is False
            assert loaded_model.batchnorm_threshold == 75

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_model_backward_compatibility(self):
        """Test loading models saved without batchnorm_threshold."""
        # Simulate an old saved model format
        model = SoundCNN(num_classes=2, dataset_size=150)

        # Create a checkpoint without batchnorm_threshold
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_classes": 2,
            "kernel_size": (3, 3),
            "use_batchnorm": True,
            "model_type": "SoundCNN"
            # Note: no batchnorm_threshold
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            torch.save(checkpoint, temp_path)

            # Should load successfully with default threshold
            device = torch.device('cpu')
            loaded_model = SoundCNN.load_model(temp_path, device)

            assert loaded_model.batchnorm_threshold == 100  # Default value
            assert loaded_model.use_batchnorm is True

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_model_training_mode_switching(self):
        """Test that model can switch between train and eval modes."""
        model = SoundCNN(num_classes=2, dataset_size=150)

        # Test train mode
        model.train()
        assert model.training is True

        # Test eval mode
        model.eval()
        assert model.training is False

    def test_model_device_movement(self):
        """Test moving model between devices."""
        model = SoundCNN(num_classes=2, dataset_size=150)

        # Move to CPU (should work regardless of CUDA availability)
        model.to(torch.device('cpu'))

        # Test forward pass still works
        x = torch.randn(1, 1, 128, 128)
        output = model(x)
        assert output.shape == (1, 2)

    def test_private_use_batchnorm_parameter(self):
        """Test the private _use_batchnorm parameter for loading."""
        # This parameter should override the automatic decision
        model = SoundCNN(num_classes=2, dataset_size=50, _use_batchnorm=True)

        # Even though dataset_size=50 < 100, _use_batchnorm=True should override
        assert model.use_batchnorm is True

        model2 = SoundCNN(num_classes=2, dataset_size=150, _use_batchnorm=False)

        # Even though dataset_size=150 >= 100, _use_batchnorm=False should override
        assert model2.use_batchnorm is False

    def test_multiple_model_instances(self):
        """Test that multiple model instances don't interfere with each other."""
        model1 = SoundCNN(num_classes=2, dataset_size=50)
        model2 = SoundCNN(num_classes=3, dataset_size=150)

        assert model1.use_batchnorm is False
        assert model2.use_batchnorm is True
        assert model1.num_classes == 2
        assert model2.num_classes == 3

        # Test that they produce different outputs
        x = torch.randn(1, 1, 128, 128)
        out1 = model1(x)
        out2 = model2(x)

        assert out1.shape == (1, 2)
        assert out2.shape == (1, 3)
        assert not torch.equal(out1[:, :2], out2[:, :2])  # Should be different


class TestSimpleCNN:
    """Test the SimpleCNN model implementation."""

    def test_basic_initialization(self):
        """Test basic SimpleCNN initialization."""
        model = SimpleCNN()

        assert model.num_classes == 2
        assert isinstance(model, AudioLoopModel)
        assert isinstance(model, torch.nn.Module)

    def test_custom_num_classes(self):
        """Test SimpleCNN with custom number of classes."""
        model = SimpleCNN(num_classes=5)

        assert model.num_classes == 5

    def test_forward_pass(self):
        """Test SimpleCNN forward pass."""
        model = SimpleCNN(num_classes=3)

        # Create a sample input (batch=2, channels=1, height=64, width=64)
        x = torch.randn(2, 1, 64, 64)
        output = model(x)

        assert output.shape == (2, 3)
        assert not torch.isnan(output).any()

    def test_model_info(self):
        """Test get_model_info method."""
        model = SimpleCNN(num_classes=4)

        info = model.get_model_info()

        assert info["model_type"] == "SimpleCNN"
        assert info["num_classes"] == 4
        assert "num_parameters" in info
        assert info["num_parameters"] > 0

    def test_save_model(self):
        """Test SimpleCNN saving functionality."""
        model = SimpleCNN(num_classes=3)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            model.save_model(temp_path)

            # Verify file exists
            assert Path(temp_path).exists()

            # Verify file contains expected data
            checkpoint = torch.load(temp_path, map_location='cpu')
            assert checkpoint["model_type"] == "SimpleCNN"
            assert checkpoint["num_classes"] == 3
            assert "model_state_dict" in checkpoint

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_model(self):
        """Test SimpleCNN loading functionality."""
        original_model = SimpleCNN(num_classes=5)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name

        try:
            # Save the model
            original_model.save_model(temp_path)

            # Load it back
            device = torch.device('cpu')
            loaded_model = SimpleCNN.load_model(temp_path, device)

            # Verify loaded model has same properties
            assert loaded_model.num_classes == original_model.num_classes

            # Verify model info matches
            original_info = original_model.get_model_info()
            loaded_info = loaded_model.get_model_info()
            assert original_info == loaded_info

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_simplecnn_vs_soundcnn_parameters(self):
        """Test that SimpleCNN has fewer parameters than SoundCNN."""
        simple_model = SimpleCNN(num_classes=2)
        sound_model = SoundCNN(num_classes=2, dataset_size=150)

        simple_params = simple_model.get_model_info()["num_parameters"]
        sound_params = sound_model.get_model_info()["num_parameters"]

        assert simple_params < sound_params

    def test_model_training_mode_switching(self):
        """Test that SimpleCNN can switch between train and eval modes."""
        model = SimpleCNN(num_classes=2)

        # Test train mode
        model.train()
        assert model.training is True

        # Test eval mode
        model.eval()
        assert model.training is False
