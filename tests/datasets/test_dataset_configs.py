"""
Tests for dataset configuration classes and the DatasetConfig ABC.

Focused on interface compliance and key behavioral differences.
"""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from audioloop.datasets.dataset_config import DatasetConfig
from audioloop.datasets.fsd50k import FSD50KConfig
from audioloop.datasets.urbansound8k import UrbanSound8KConfig


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

            def get_metadata_entries(self):
                return []

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
                return True

        config = CompleteConfig()
        assert isinstance(config, DatasetConfig)

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_concrete_classes_implement_interface(self, config_class):
        """Test that concrete classes implement required interface."""
        config = config_class()

        # Test required methods exist and are callable
        assert hasattr(config, "get_metadata_entries")
        assert hasattr(config, "get_audio_path")
        assert callable(config.get_metadata_entries)
        assert callable(config.get_audio_path)


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
    """Test metadata entry generation with minimal mocking."""

    @patch(
        "builtins.open", new_callable=mock_open, read_data="fname\tlabels\tmids\n100032\t1,5\tm1,m5"
    )
    @patch("audioloop.datasets.fsd50k.load_fsd50k_vocabulary")
    def test_fsd50k_metadata_parsing(self, mock_vocab, mock_file):
        """Test FSD50K metadata entry generation."""
        mock_vocab.return_value = {1: "Drill", 5: "Music"}

        config = FSD50KConfig()
        entries = config.get_metadata_entries()

        # Should create entry for each label
        assert len(entries) == 2
        assert entries[0]["filename"] == "100032"
        assert entries[0]["class_name"] in ["Drill", "Music"]
        assert entries[0]["fold"] is None

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="slice_file_name,fold,classID,class\n7061-6-0-0.wav,5,0,air_conditioner",
    )
    @patch("audioloop.datasets.urbansound8k.load_urbansound8k_vocabulary")
    def test_urbansound8k_metadata_parsing(self, mock_vocab, mock_file):
        """Test UrbanSound8K metadata entry generation."""
        mock_vocab.return_value = {0: "air_conditioner"}

        config = UrbanSound8KConfig()
        entries = config.get_metadata_entries()

        assert len(entries) == 1
        assert entries[0]["filename"] == "7061-6-0-0.wav"
        assert entries[0]["class_name"] == "air_conditioner"
        assert entries[0]["fold"] == 5

    @pytest.mark.parametrize("config_class", [FSD50KConfig, UrbanSound8KConfig])
    def test_metadata_entries_return_list(self, config_class):
        """Test that get_metadata_entries returns a list."""
        config = config_class()

        # Mock file operations to avoid actual file access
        with patch("builtins.open", mock_open(read_data="")):
            # Mock vocabulary loading
            if config_class == FSD50KConfig:
                with patch("audioloop.datasets.fsd50k.load_fsd50k_vocabulary", return_value={}):
                    entries = config.get_metadata_entries()
            else:
                with patch(
                    "audioloop.datasets.urbansound8k.load_urbansound8k_vocabulary", return_value={}
                ):
                    entries = config.get_metadata_entries()

            assert isinstance(entries, list)


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
