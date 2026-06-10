"""Tests for the AudioLoop model system."""

import pytest
import torch

from audioloop.active_learning_core import load_model
from audioloop.models import AudioLoopModel
from audioloop.models.cnn5layer import CNN5Layer
from audioloop.models.cnn7layer import CNN7Layer
from audioloop.models.simplecnn import SimpleCnn


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

        class CompleteModel(AudioLoopModel):
            description = "Test model for unit tests"

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x)

            def get_model_info(self) -> dict:
                return {"model_type": "test", "num_classes": 1, "num_parameters": 11}

            def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
                return True  # Test model accepts any shape

        # Should not raise an exception
        model = CompleteModel()
        assert isinstance(model, AudioLoopModel)
        assert isinstance(model, torch.nn.Module)


class TestCNN5Layer:
    """Test the CNN5Layer model implementation."""

    def test_basic_initialization(self):
        """Test basic CNN5Layer initialization."""
        model = CNN5Layer(num_classes=2)

        assert model.num_classes == 2
        assert model.kernel_size == (3, 3)
        assert model.batchnorm_threshold == 100
        assert model.use_batchnorm is True  # Default when no dataset_size
        assert isinstance(model, AudioLoopModel)
        assert isinstance(model, torch.nn.Module)

    def test_batchnorm_decision_large_dataset(self):
        """Test BatchNorm decision for large datasets."""
        model = CNN5Layer(num_classes=2, dataset_size=150)

        assert model.use_batchnorm is True
        assert hasattr(model, "bn1")
        assert hasattr(model, "bn2")

    def test_batchnorm_decision_small_dataset(self):
        """Test BatchNorm decision for small datasets."""
        model = CNN5Layer(num_classes=2, dataset_size=50)

        assert model.use_batchnorm is False
        assert not hasattr(model, "bn1")
        assert not hasattr(model, "bn2")
        assert hasattr(model, "dropout")  # Should have dropout instead

    def test_batchnorm_decision_at_threshold(self):
        """Test BatchNorm decision exactly at threshold."""
        model = CNN5Layer(num_classes=2, dataset_size=100)

        assert model.use_batchnorm is True

        model_below = CNN5Layer(num_classes=2, dataset_size=99)
        assert model_below.use_batchnorm is False

    def test_custom_batchnorm_threshold(self):
        """Test custom BatchNorm threshold."""
        model = CNN5Layer(num_classes=2, dataset_size=50, batchnorm_threshold=40)

        assert model.use_batchnorm is True
        assert model.batchnorm_threshold == 40

    def test_custom_threshold_extreme_values(self):
        """Test extreme threshold values."""
        # Always use BatchNorm
        model_always = CNN5Layer(num_classes=2, dataset_size=1, batchnorm_threshold=0)
        assert model_always.use_batchnorm is True

        # Never use BatchNorm
        model_never = CNN5Layer(num_classes=2, dataset_size=1000, batchnorm_threshold=float("inf"))
        assert model_never.use_batchnorm is False

    def test_no_dataset_size_defaults_to_batchnorm(self):
        """Test that missing dataset_size defaults to using BatchNorm."""
        model = CNN5Layer(num_classes=2)

        assert model.use_batchnorm is True

    def test_custom_parameters(self):
        """Test custom kernel size and other parameters."""
        model = CNN5Layer(
            num_classes=5, kernel_size=(5, 5), dataset_size=200, batchnorm_threshold=150
        )

        assert model.num_classes == 5
        assert model.kernel_size == (5, 5)
        assert model.batchnorm_threshold == 150
        assert model.use_batchnorm is True

    def test_forward_pass_with_batchnorm(self):
        """Test forward pass with BatchNorm enabled."""
        model = CNN5Layer(num_classes=2, dataset_size=150)

        # Create a sample input (batch=1, channels=1, height=128, width=128)
        x = torch.randn(1, 1, 128, 128)
        output = model(x)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_forward_pass_without_batchnorm(self):
        """Test forward pass without BatchNorm."""
        model = CNN5Layer(num_classes=2, dataset_size=50)

        # Create a sample input
        x = torch.randn(1, 1, 128, 128)
        output = model(x)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_model_info(self):
        """Test get_model_info method."""
        model = CNN5Layer(num_classes=3, dataset_size=75, batchnorm_threshold=80)

        info = model.get_model_info()

        assert info["model_type"] == "cnn5layer"
        assert info["num_classes"] == 3
        assert info["kernel_size"] == (3, 3)
        assert info["use_batchnorm"] is False  # 75 < 80
        assert info["batchnorm_threshold"] == 80
        assert "num_parameters" in info
        assert info["num_parameters"] > 0

    def test_parameter_count_difference(self):
        """Test that BatchNorm affects parameter count."""
        model_with_bn = CNN5Layer(num_classes=2, dataset_size=150)
        model_without_bn = CNN5Layer(num_classes=2, dataset_size=50)

        params_with_bn = model_with_bn.get_model_info()["num_parameters"]
        params_without_bn = model_without_bn.get_model_info()["num_parameters"]

        assert params_with_bn > params_without_bn

    def test_can_handle_shape(self):
        """Test can_handle_shape method."""
        model = CNN5Layer(num_classes=2, dataset_size=150)

        # Should accept 2D tensors
        assert model.can_handle_shape((128, 993)) is True
        assert model.can_handle_shape((64, 1500)) is True
        assert model.can_handle_shape((256, 500)) is True

        # Should reject non-2D tensors
        assert model.can_handle_shape((128,)) is False  # 1D
        assert model.can_handle_shape((128, 993, 3)) is False  # 3D

    def test_explicit_use_batchnorm_parameter(self):
        """Test the explicit use_batchnorm parameter override."""
        # This parameter should override the automatic decision
        model = CNN5Layer(num_classes=2, dataset_size=50, use_batchnorm=True)

        # Even though dataset_size=50 < 100, use_batchnorm=True should override
        assert model.use_batchnorm is True

        model2 = CNN5Layer(num_classes=2, dataset_size=150, use_batchnorm=False)

        # Even though dataset_size=150 >= 100, use_batchnorm=False should override
        assert model2.use_batchnorm is False

    def test_multiple_model_instances(self):
        """Test that multiple model instances don't interfere with each other."""
        model1 = CNN5Layer(num_classes=2, dataset_size=50)
        model2 = CNN5Layer(num_classes=3, dataset_size=150)

        assert model1.use_batchnorm is False
        assert model2.use_batchnorm is True
        assert model1.num_classes == 2
        assert model2.num_classes == 3

        # Test that they produce different outputs
        x = torch.randn(1, 1, 128, 128)
        output1 = model1(x)
        output2 = model2(x)

        assert output1.shape == (1, 2)
        assert output2.shape == (1, 3)
        assert not torch.equal(output1[:, :2], output2[:, :2])  # Should be different


class TestCNN7Layer:
    """Forward-pass tests for CNN7Layer.

    The checkpoint round-trip below covers its constructor and state dict, but until these
    tests nothing ever pushed a tensor through the 7-layer forward — the only registered
    model whose forward was untested.
    """

    def test_forward_pass_with_batchnorm(self):
        model = CNN7Layer(num_classes=2, dataset_size=150)

        output = model(torch.randn(1, 1, 128, 128))

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_forward_pass_without_batchnorm_uses_dropout_path(self):
        model = CNN7Layer(num_classes=3, dataset_size=50)
        assert model.use_batchnorm is False

        output = model(torch.randn(2, 1, 128, 128))

        assert output.shape == (2, 3)
        assert not torch.isnan(output).any()

    def test_forward_adds_channel_dim_for_3d_input(self):
        model = CNN7Layer(num_classes=2, dataset_size=150)

        output = model(torch.randn(1, 128, 128))  # no channel dim

        assert output.shape == (1, 2)

    def test_can_handle_shape(self):
        model = CNN7Layer(num_classes=2, dataset_size=150)

        assert model.can_handle_shape((128, 993)) is True
        assert model.can_handle_shape((128, -1)) is True  # variable time dim
        assert model.can_handle_shape((128,)) is False
        assert model.can_handle_shape((128, 993, 3)) is False


class TestModelCheckpointMetadata:
    """Test model checkpoint metadata round-trips through inference loading."""

    @pytest.mark.parametrize(
        ("model_type", "model_class", "init_kwargs"),
        [
            ("cnn5layer", CNN5Layer, {"num_classes": 2, "dataset_size": 50}),
            ("cnn7layer", CNN7Layer, {"num_classes": 3, "dataset_size": 150}),
            ("simplecnn", SimpleCnn, {"num_classes": 4}),
        ],
    )
    def test_builtin_model_checkpoint_round_trip(
        self, tmp_path, model_type, model_class, init_kwargs
    ):
        """Test that built-in model metadata can reconstruct saved checkpoints."""
        model = model_class(**init_kwargs)
        model.eval()

        model_info = model.get_model_info()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            **{k: v for k, v in model_info.items() if k != "num_parameters"},
        }
        checkpoint_path = tmp_path / f"{model_type}.pt"
        torch.save(checkpoint, checkpoint_path)

        loaded_model = load_model(checkpoint_path, torch.device("cpu"))

        assert isinstance(loaded_model, model_class)
        assert loaded_model.training is False
        assert loaded_model.get_model_info()["model_type"] == model_type
        assert loaded_model.get_model_info()["num_classes"] == init_kwargs["num_classes"]

        if "use_batchnorm" in model_info:
            assert loaded_model.get_model_info()["use_batchnorm"] == model_info["use_batchnorm"]

        for name, parameter in model.state_dict().items():
            assert torch.equal(parameter, loaded_model.state_dict()[name])


class TestSimpleCnn:
    """Test the SimpleCnn model implementation."""

    def test_basic_initialization(self):
        """Test basic SimpleCnn initialization."""
        model = SimpleCnn()

        assert model.num_classes == 2
        assert isinstance(model, AudioLoopModel)
        assert isinstance(model, torch.nn.Module)

    def test_custom_num_classes(self):
        """Test SimpleCnn with custom number of classes."""
        model = SimpleCnn(num_classes=5)

        assert model.num_classes == 5

    def test_forward_pass(self):
        """Test SimpleCnn forward pass."""
        model = SimpleCnn(num_classes=3)

        # Create a sample input (batch=2, channels=1, height=64, width=64)
        x = torch.randn(2, 1, 64, 64)
        output = model(x)

        assert output.shape == (2, 3)
        assert not torch.isnan(output).any()

    def test_model_info(self):
        """Test get_model_info method."""
        model = SimpleCnn(num_classes=4)

        info = model.get_model_info()

        assert info["model_type"] == "simplecnn"
        assert info["num_classes"] == 4
        assert "num_parameters" in info
        assert info["num_parameters"] > 0

    def test_can_handle_shape(self):
        """Test can_handle_shape method."""
        model = SimpleCnn(num_classes=2)

        # Should accept 2D tensors
        assert model.can_handle_shape((128, 993)) is True
        assert model.can_handle_shape((64, 1500)) is True
        assert model.can_handle_shape((256, 500)) is True

        # Should reject non-2D tensors
        assert model.can_handle_shape((128,)) is False  # 1D
        assert model.can_handle_shape((128, 993, 3)) is False  # 3D

    def test_simplecnn_vs_cnn5layer_parameters(self):
        """Test that SimpleCnn has fewer parameters than CNN5Layer."""
        simple_model = SimpleCnn(num_classes=2)
        sound_model = CNN5Layer(num_classes=2, dataset_size=150)

        simple_params = simple_model.get_model_info()["num_parameters"]
        sound_params = sound_model.get_model_info()["num_parameters"]

        assert simple_params < sound_params
