"""
Tests for dataset configuration classes and the DatasetConfig ABC.

Focused on interface compliance and key behavioral differences.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from audioloop.datasets.dataset_config import DatasetConfig
from audioloop.datasets.fsd50k_config import FSD50KConfig
from audioloop.datasets.urbansound8k_config import UrbanSound8KConfig


class TestDatasetConfigInterface:
    """Test the DatasetConfig abstract base class interface."""

    def test_cannot_instantiate_abc(self):
        """Test that DatasetConfig cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DatasetConfig()  # type: ignore[abstract]

    def test_subclass_must_implement_methods(self):
        """Test that incomplete subclasses cannot be instantiated."""

        class IncompleteConfig(DatasetConfig):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteConfig()  # type: ignore[abstract]

    def test_complete_subclass_works(self):
        """Test that complete implementation can be instantiated."""

        class CompleteConfig(DatasetConfig):
            @property
            def dataset_csv(self) -> Path:
                return Path("test.csv")

            @property
            def audio_root(self) -> Path:
                return Path("test_audio")

            @property
            def name_to_id(self) -> dict[str, int]:
                return {"test": 0}

            @property
            def vocabulary(self) -> dict[int, str]:
                return {0: "test"}

            def load_metadata(self, split="dev"):
                return []

            def list_classes(self):
                print("test class")

            def get_audio_path(self, filename, fold=None):
                return Path(filename)

            def get_audio_processing_params(self):
                return {}

            def is_positive_class(self, class_name, positive_class):
                return class_name == positive_class

            def get_spectrogram_path(self, filename, specs_dir):
                return specs_dir / f"{filename}.pt"

            def create_spectrogram_transform(self):
                import torch.nn as nn

                return nn.Sequential()

            def parse_metadata_row(self, row):
                return row

            def get_binary_label(self, item, positive_class_id, positive_class_name):
                return 1

            def fix_spectrogram_length(self, spec):
                return spec

            def get_output_shape(self) -> tuple[int, ...]:
                return (128, 993)  # Test shape

            def process_single_file(self, file_info, output_dir):
                return True, None

        config = CompleteConfig()
        assert isinstance(config, DatasetConfig)

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_concrete_classes_implement_interface(self, config_class):
        """Test that concrete classes implement required interface."""
        config = config_class()

        # Test required methods exist and are callable
        assert hasattr(config, "load_metadata")
        assert hasattr(config, "get_audio_path")
        assert hasattr(config, "fix_spectrogram_length")
        assert hasattr(config, "process_single_file")
        assert hasattr(config, "get_output_shape")
        assert callable(config.load_metadata)
        assert callable(config.get_audio_path)
        assert callable(config.fix_spectrogram_length)
        assert callable(config.process_single_file)
        assert callable(config.get_output_shape)


class TestDatasetConfigBehavior:
    """Test behavioral differences between dataset configurations."""

    def test_fsd50k_ignores_fold_parameter(self):
        """Test that FSD50K ignores fold parameter in get_audio_path."""
        config = FSD50KConfig()
        path_without_fold = config.get_audio_path("test.wav")
        path_with_fold = config.get_audio_path("test.wav", fold=5)

        # Should be identical (fold ignored)
        assert path_without_fold == path_with_fold
        assert "fold" not in str(path_without_fold)

    def test_urbansound8k_uses_fold_parameter(self):
        """Test that UrbanSound8K uses fold parameter in get_audio_path."""
        config = UrbanSound8KConfig()
        path = config.get_audio_path("test.wav", fold=3)

        # Should include fold in path
        assert "fold3" in str(path)

    def test_different_fixed_lengths(self):
        """Test that datasets have different fixed lengths for spectrograms."""
        fsd50k = FSD50KConfig()
        urbansound8k = UrbanSound8KConfig()

        # Both return variable time dimension (-1) in output shape
        assert fsd50k.get_output_shape()[1] == -1
        assert urbansound8k.get_output_shape()[1] == -1

        # But still have different max length limits for outlier handling
        assert fsd50k.fixed_length != urbansound8k.fixed_length
        assert fsd50k.fixed_length == 2048
        assert urbansound8k.fixed_length == 993

    def test_consistent_audio_parameters(self):
        """Test that datasets use consistent audio processing parameters."""
        fsd50k = FSD50KConfig()
        urbansound8k = UrbanSound8KConfig()

        # These should be the same for consistency
        assert fsd50k.sample_rate == urbansound8k.sample_rate
        assert fsd50k.n_fft == urbansound8k.n_fft
        assert fsd50k.hop_length == urbansound8k.hop_length
        assert fsd50k.n_mels == urbansound8k.n_mels


class TestMetadataHandling:
    """Test basic metadata functionality."""

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_load_metadata_returns_list(self, config_class):
        """Test that load_metadata returns a list using test fixtures."""
        # Get the test fixtures directory
        test_dir = Path(__file__).parent.parent / "fixtures"

        if config_class == FSD50KConfig:
            # Use test fixtures for FSD50K
            config = config_class()
            # Override the CSV paths to point to test fixtures
            config.dev_csv = test_dir / "fsd50k" / "dev.csv"
            config.eval_csv = test_dir / "fsd50k" / "eval.csv"

            with patch("audioloop.datasets.fsd50k_config.load_fsd50k_vocabulary", return_value={}):
                entries = config.load_metadata()
        else:
            # Use test fixtures for UrbanSound8K
            config = config_class()
            # Override the CSV path to point to test fixtures
            config.metadata_csv = test_dir / "urbansound8k" / "UrbanSound8K.csv"

            with patch(
                "audioloop.datasets.urbansound8k_config.load_urbansound8k_vocabulary",
                return_value={},
            ):
                entries = config.load_metadata()

        assert isinstance(entries, list)
        assert len(entries) > 0  # Should have some test data


class TestNewAbstractMethods:
    """Test the newly added abstract methods."""

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_fix_spectrogram_length(self, config_class):
        """Test that fix_spectrogram_length works correctly with new variable length behavior."""
        import torch

        config = config_class()

        # Test with a short spectrogram (should be preserved, no padding)
        short_spec = torch.randn(1, 128, 100)
        fixed_spec = config.fix_spectrogram_length(short_spec)

        # Should preserve natural length (no padding)
        assert fixed_spec.shape[-1] == 100

        # Test with a spectrogram within reasonable limits (should be preserved)
        medium_spec = torch.randn(1, 128, config.fixed_length // 2)
        fixed_spec = config.fix_spectrogram_length(medium_spec)

        # Should preserve natural length
        assert fixed_spec.shape[-1] == config.fixed_length // 2

        # Test with an outlier spectrogram (should be cropped)
        outlier_spec = torch.randn(1, 128, config.fixed_length * 2)
        fixed_spec = config.fix_spectrogram_length(outlier_spec)

        # Should be cropped to max allowed length
        assert fixed_spec.shape[-1] == config.fixed_length

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_process_single_file_signature(self, config_class):
        """Test that process_single_file has correct signature."""
        config = config_class()

        # Should have the method (don't actually call it - might need real files)
        assert hasattr(config, "process_single_file")
        assert callable(config.process_single_file)

        # Just verify it's callable - actual testing would need file mocking

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_get_output_shape(self, config_class):
        """Test that get_output_shape returns correct shape."""
        config = config_class()

        shape = config.get_output_shape()

        # Should return a tuple of integers
        assert isinstance(shape, tuple)
        assert len(shape) == 2  # Should be 2D for spectrograms
        assert all(isinstance(dim, int) for dim in shape)

        # Should match expected dimensions
        assert shape[0] == config.n_mels  # Frequency dimension
        assert shape[1] == -1  # Time dimension is variable (-1 sentinel)


class TestPathGeneration:
    """Test audio path generation patterns."""

    def test_fsd50k_path_structure(self):
        """Test FSD50K path generation structure."""
        config = FSD50KConfig()
        path = config.get_audio_path("100032.wav")

        # Should be simple: audio_root/filename
        expected = config.audio_root / "100032.wav"
        assert path == expected

    def test_urbansound8k_path_structure(self):
        """Test UrbanSound8K path generation structure."""
        config = UrbanSound8KConfig()
        path = config.get_audio_path("test.wav", fold=3)

        # Should include fold: audio_root/fold3/filename
        expected = config.audio_root / "fold3" / "test.wav"
        assert path == expected

    def test_urbansound8k_fallback_behavior(self):
        """Test UrbanSound8K fallback when no fold specified."""
        config = UrbanSound8KConfig()

        # Mock path existence to test fallback logic
        with patch.object(Path, "exists", return_value=False):
            path = config.get_audio_path("test.wav")
            # Should default to fold1 when not found
            assert "fold1" in str(path)
