"""
Tests for dataset configuration classes and the DatasetConfig ABC.

Focused on interface compliance and key behavioral differences.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from audioloop.datasets.audioset_config import AudiosetConfig
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
            def audio_extension(self) -> str:
                return ".wav"

            @property
            def name_to_id(self) -> dict[str, int]:
                return {"test": 0}

            @property
            def vocabulary(self) -> dict[int, str]:
                return {0: "test"}

            def get_available_splits(self) -> list[str]:
                return ["test"]

            def get_default_split(self) -> str:
                return "test"

            def _load_metadata_for_split(self, split: str):
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

            def parse_metadata_row(self, row, split=None):
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
        assert hasattr(config, "get_available_splits")
        assert hasattr(config, "get_default_split")
        assert hasattr(config, "get_audio_path")
        assert hasattr(config, "fix_spectrogram_length")
        assert hasattr(config, "process_single_file")
        assert hasattr(config, "get_output_shape")
        assert callable(config.load_metadata)
        assert callable(config.get_available_splits)
        assert callable(config.get_default_split)
        assert callable(config.get_audio_path)
        assert callable(config.fix_spectrogram_length)
        assert callable(config.process_single_file)
        assert callable(config.get_output_shape)


class TestDatasetSplitInterface:
    """Test the new split validation interface."""

    @pytest.mark.parametrize(
        "config_class,expected_splits",
        [
            (FSD50KConfig, ["dev", "eval"]),
            (UrbanSound8KConfig, ["all"]),
        ],
    )
    def test_get_available_splits(self, config_class, expected_splits):
        """Test that datasets return correct available splits."""
        config = config_class()
        splits = config.get_available_splits()
        assert splits == expected_splits

    @pytest.mark.parametrize(
        "config_class,expected_default",
        [
            (FSD50KConfig, "dev"),
            (UrbanSound8KConfig, "all"),
        ],
    )
    def test_get_default_split(self, config_class, expected_default):
        """Test that datasets return correct default splits."""
        config = config_class()
        default = config.get_default_split()
        assert default == expected_default

    def test_load_metadata_with_default_split(self):
        """Test that load_metadata() with no args uses default split."""
        config = FSD50KConfig()

        # Mock file system to avoid needing actual files
        with patch.object(
            config, "_load_metadata_for_split", return_value=[{"test": "data"}]
        ) as mock_load:
            result = config.load_metadata()
            mock_load.assert_called_once_with("dev")  # Should use default
            assert result == [{"test": "data"}]

    def test_load_metadata_with_explicit_valid_split(self):
        """Test that load_metadata() with valid split works."""
        config = FSD50KConfig()

        with patch.object(
            config, "_load_metadata_for_split", return_value=[{"eval": "data"}]
        ) as mock_load:
            result = config.load_metadata(split="eval")
            mock_load.assert_called_once_with("eval")
            assert result == [{"eval": "data"}]

    def test_load_metadata_with_invalid_split_raises_error(self):
        """Test that load_metadata() with invalid split raises clear error."""
        config = FSD50KConfig()

        with pytest.raises(ValueError, match="Split 'invalid' not available for FSD50KConfig"):
            config.load_metadata(split="invalid")

    def test_urbansound8k_rejects_dev_split(self):
        """Test that UrbanSound8K correctly rejects 'dev' split."""
        config = UrbanSound8KConfig()

        with pytest.raises(ValueError, match="Split 'dev' not available for UrbanSound8KConfig"):
            config.load_metadata(split="dev")

    def test_error_message_shows_valid_splits(self):
        """Test that error message includes list of valid splits."""
        config = UrbanSound8KConfig()

        with pytest.raises(ValueError, match=r"Valid splits: \['all'\]"):
            config.load_metadata(split="nonexistent")

    def test_default_split_is_in_available_splits(self):
        """Test that default split is always in available splits list."""
        for config_class in [FSD50KConfig, UrbanSound8KConfig]:
            config = config_class()
            default = config.get_default_split()
            available = config.get_available_splits()
            assert default in available, (
                f"{config_class.__name__} default '{default}' not in available {available}"
            )


class TestDatasetConfigBehavior:
    """Test behavioral differences between dataset configurations."""

    def test_fsd50k_uses_split_parameter(self):
        """Test that FSD50K uses split parameter in get_audio_path."""
        config = FSD50KConfig()
        dev_path = config.get_audio_path("test.wav", split="dev")
        eval_path = config.get_audio_path("test.wav", split="eval")

        # Should include split in path
        assert "dev_audio" in str(dev_path)
        assert "eval_audio" in str(eval_path)
        assert dev_path != eval_path

        # Should ignore fold parameter
        path_without_fold = config.get_audio_path("test.wav", split="dev")
        path_with_fold = config.get_audio_path("test.wav", split="dev", fold=5)
        assert path_without_fold == path_with_fold

    def test_urbansound8k_uses_fold_parameter(self):
        """Test that UrbanSound8K uses fold parameter in get_audio_path."""
        config = UrbanSound8KConfig()
        path = config.get_audio_path("test.wav", fold=3)

        # Should include fold in path
        assert "fold3" in str(path)

    def test_different_max_spectrogram_lengths(self):
        """Test that datasets have different fixed lengths for spectrograms."""
        fsd50k = FSD50KConfig()
        urbansound8k = UrbanSound8KConfig()

        # Both return variable time dimension (-1) in output shape
        assert fsd50k.get_output_shape()[1] == -1
        assert urbansound8k.get_output_shape()[1] == -1

        # But still have different max length limits for outlier handling
        assert fsd50k._max_spectrogram_length != urbansound8k._max_spectrogram_length
        assert fsd50k._max_spectrogram_length == 2048
        assert urbansound8k._max_spectrogram_length == 993

    def test_consistent_audio_parameters(self):
        """Test that datasets use consistent audio processing parameters."""
        fsd50k = FSD50KConfig()
        urbansound8k = UrbanSound8KConfig()

        # These should be the same for consistency
        assert fsd50k._sample_rate == urbansound8k._sample_rate
        assert fsd50k._n_fft == urbansound8k._n_fft
        assert fsd50k._hop_length == urbansound8k._hop_length
        assert fsd50k._n_mels == urbansound8k._n_mels


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
        medium_spec = torch.randn(1, 128, config._max_spectrogram_length // 2)
        fixed_spec = config.fix_spectrogram_length(medium_spec)

        # Should preserve natural length
        assert fixed_spec.shape[-1] == config._max_spectrogram_length // 2

        # Test with an outlier spectrogram (should be cropped)
        outlier_spec = torch.randn(1, 128, config._max_spectrogram_length * 2)
        fixed_spec = config.fix_spectrogram_length(outlier_spec)

        # Should be cropped to max allowed length
        assert fixed_spec.shape[-1] == config._max_spectrogram_length

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
        assert shape[0] == config._n_mels  # Frequency dimension
        assert shape[1] == -1  # Time dimension is variable (-1 sentinel)


class TestPathGeneration:
    """Test audio path generation patterns."""

    def test_fsd50k_path_structure(self):
        """Test FSD50K path generation structure."""
        config = FSD50KConfig()
        path = config.get_audio_path("100032.wav", split="dev")

        # Should include split subdirectory: audio_root/FSD50K.dev_audio/filename
        expected = config.audio_root / "FSD50K.dev_audio" / "100032.wav"
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


class TestAudioSetConfig:
    """Test AudioSet-specific configuration behavior."""

    def test_audioset_splits(self):
        """Test that AudioSet returns correct available splits."""
        config = AudiosetConfig()
        splits = config.get_available_splits()

        assert "bal_train" in splits
        assert "unbal_train" in splits
        assert "eval" in splits
        assert len(splits) == 3

    def test_audioset_default_split(self):
        """Test that AudioSet uses unbal_train as default."""
        config = AudiosetConfig()
        assert config.get_default_split() == "unbal_train"

    def test_audioset_path_structure(self):
        """Test AudioSet path generation with subdirectories."""
        config = AudiosetConfig()

        # AudioSet organizes files by first 2 characters
        path = config.get_audio_path("abc123.flac", split="bal_train")

        # Should be: audio_root/bal_train/ab/abc123.flac
        assert "bal_train" in str(path)
        assert "/ab/" in str(path)
        assert "abc123.flac" in str(path)

    def test_audioset_path_split_parameter(self):
        """Test that AudioSet correctly uses split parameter."""
        config = AudiosetConfig()

        bal_path = config.get_audio_path("test.flac", split="bal_train")
        eval_path = config.get_audio_path("test.flac", split="eval")

        assert "bal_train" in str(bal_path)
        assert "eval" in str(eval_path)
        assert bal_path != eval_path

    def test_audioset_path_fallback_to_unbal_train(self):
        """Test that AudioSet falls back to unbal_train when split not provided."""
        config = AudiosetConfig()

        # No split provided - should default to unbal_train
        path = config.get_audio_path("test.flac")
        assert "unbal_train" in str(path)

    def test_audioset_parse_metadata_includes_split(self):
        """Test that parsed metadata includes split information."""
        config = AudiosetConfig()

        # Mock the ontology loading
        config._mid_to_name = {"/m/test": "Test Class"}

        row = {
            "YTID": "abc123",
            "start_seconds": "1.5",
            "end_seconds": "11.5",
            "positive_labels": '"/m/test"',
        }

        parsed = config.parse_metadata_row(row, split="bal_train")

        assert parsed["split"] == "bal_train"
        assert parsed["filename"] == "abc123.flac"
        assert parsed["ytid"] == "abc123"
        assert "Test Class" in parsed["labels"]

    def test_audioset_parse_metadata_preserves_split_for_audio_path(self):
        """Test that split is preserved in metadata for correct audio path resolution."""
        config = AudiosetConfig()

        # Mock ontology
        config._mid_to_name = {"/m/test": "Test"}

        row = {
            "YTID": "YT123",
            "start_seconds": "0.0",
            "end_seconds": "10.0",
            "positive_labels": '"/m/test"',
        }

        # Parse with different splits
        parsed_bal = config.parse_metadata_row(row, split="bal_train")
        parsed_unbal = config.parse_metadata_row(row, split="unbal_train")
        parsed_eval = config.parse_metadata_row(row, split="eval")

        # Audio paths should differ based on split
        assert "bal_train" in str(parsed_bal["audio_path"])
        assert "unbal_train" in str(parsed_unbal["audio_path"])
        assert "eval" in str(parsed_eval["audio_path"])

        # But filename should be the same
        assert parsed_bal["filename"] == parsed_unbal["filename"] == parsed_eval["filename"]

    def test_audioset_load_metadata_passes_split_to_parser(self):
        """Test that load_metadata passes correct split to parse_metadata_row."""
        config = AudiosetConfig()

        # Create minimal CSV content
        csv_content = '# YTID,start_seconds,end_seconds,positive_labels\nYT123,0.0,10.0,"/m/test"\n'

        # Mock ontology
        config._mid_to_name = {"/m/test": "Test"}
        config._name_to_mid = {"Test": 0}  # Integer ID mapping

        # Mock the CSV file reading for different splits
        with patch("builtins.open", create=True) as mock_open:
            from io import StringIO

            mock_open.return_value.__enter__.return_value = StringIO(csv_content)

            # Temporarily override CSV paths
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                tmp.write(csv_content)
                tmp_path = Path(tmp.name)

            try:
                config.balanced_csv = tmp_path
                config.unbalanced_csv = tmp_path
                config.eval_csv = tmp_path

                # Load from different splits
                bal_metadata = config.load_metadata(split="bal_train")
                unbal_metadata = config.load_metadata(split="unbal_train")
                eval_metadata = config.load_metadata(split="eval")

                # Verify split is in metadata
                assert bal_metadata[0]["split"] == "bal_train"
                assert unbal_metadata[0]["split"] == "unbal_train"
                assert eval_metadata[0]["split"] == "eval"

                # Verify audio paths use correct split
                assert "bal_train" in str(bal_metadata[0]["audio_path"])
                assert "unbal_train" in str(unbal_metadata[0]["audio_path"])
                assert "eval" in str(eval_metadata[0]["audio_path"])
            finally:
                tmp_path.unlink()

    def test_audioset_supports_custom_csv(self):
        """Test that AudioSet supports custom CSV files."""
        config = AudiosetConfig()
        assert config.supports_custom_csv() is True

    def test_audioset_create_subset_raises_not_implemented_error(self):
        """Test that create_subset is implemented for AudioSet."""
        config = AudiosetConfig()

        # create_subset should be implemented (not raise NotImplementedError)
        assert hasattr(config, "create_subset")
        assert callable(config.create_subset)

    def test_audioset_output_shape(self):
        """Test that AudioSet returns correct output shape."""
        config = AudiosetConfig()
        shape = config.get_output_shape()

        # Should be (n_mels, -1) for variable time dimension
        assert shape[0] == config._n_mels
        assert shape[1] == -1


class TestCreateSubsetInterface:
    """Test the create_subset interface across datasets."""

    def test_create_subset_default_not_implemented(self):
        """Test that create_subset raises NotImplementedError by default."""

        # Create a minimal dataset config
        class MinimalConfig(DatasetConfig):
            @property
            def dataset_csv(self) -> Path:
                return Path("test.csv")

            @property
            def audio_root(self) -> Path:
                return Path("test_audio")

            @property
            def audio_extension(self) -> str:
                return ".wav"

            @property
            def name_to_id(self) -> dict[str, int]:
                return {"test": 0}

            @property
            def vocabulary(self) -> dict[int, str]:
                return {0: "test"}

            def get_available_splits(self) -> list[str]:
                return ["test"]

            def get_default_split(self) -> str:
                return "test"

            def _load_metadata_for_split(self, split: str):
                return []

            def list_classes(self):
                print("test")

            def get_audio_path(self, filename, fold=None):
                return Path(filename)

            def get_audio_processing_params(self):
                return {}

            def is_positive_class(self, class_name, positive_class):
                return False

            def get_spectrogram_path(self, filename, specs_dir):
                return specs_dir / f"{filename}.pt"

            def create_spectrogram_transform(self):
                import torch.nn as nn

                return nn.Sequential()

            def parse_metadata_row(self, row, split=None):
                return row

            def get_binary_label(self, item, positive_class_id, positive_class_name):
                return 0

            def fix_spectrogram_length(self, spec):
                return spec

            def get_output_shape(self) -> tuple[int, ...]:
                return (128, -1)

            def process_single_file(self, file_info, output_dir):
                return True, None

        config = MinimalConfig()

        with pytest.raises(NotImplementedError, match="doesn't support create_subset"):
            config.create_subset(output_path=Path("test.csv"), class_name="test", max_samples=100)

    def test_audioset_implements_create_subset(self):
        """Test that AudioSet implements create_subset."""
        config = AudiosetConfig()

        # Should have the method implemented (not just inherited)
        assert "create_subset" in dir(AudiosetConfig)

        # The method should be callable
        assert callable(config.create_subset)

    def test_audioset_create_subset_basic_functionality(self, tmp_path):
        """Test basic create_subset functionality with mocked metadata."""
        config = AudiosetConfig()

        # Mock the ontology and metadata
        config._mid_to_name = {"/m/dog": "Dog", "/m/cat": "Cat"}
        config._name_to_mid = {"Dog": 0, "Cat": 1}  # Integer ID mapping

        # Mock load_metadata to return fake data
        fake_metadata = [
            {
                "filename": "dog1.flac",
                "labels": ["Dog"],
                "split": "unbal_train",
                "audio_path": Path("/fake/dog1.flac"),
            },
            {
                "filename": "dog2.flac",
                "labels": ["Dog"],
                "split": "unbal_train",
                "audio_path": Path("/fake/dog2.flac"),
            },
            {
                "filename": "cat1.flac",
                "labels": ["Cat"],
                "split": "unbal_train",
                "audio_path": Path("/fake/cat1.flac"),
            },
            {
                "filename": "cat2.flac",
                "labels": ["Cat"],
                "split": "unbal_train",
                "audio_path": Path("/fake/cat2.flac"),
            },
        ]

        with patch.object(config, "load_metadata", return_value=fake_metadata):
            output_path = tmp_path / "test_subset.csv"
            result_path = config.create_subset(
                output_path=output_path,
                class_name="Dog",
                max_samples=3,
                positive_ratio=0.5,
                seed=42,
            )

            # Should return the output path
            assert result_path == output_path

            # File should exist
            assert output_path.exists()

            # Read and verify CSV contents
            import csv

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # Should have requested samples
            assert len(rows) <= 3

            # Should have expected columns
            assert "filename" in rows[0]
            assert "label" in rows[0]
            assert "original_class" in rows[0]
            assert "split" in rows[0]
            assert "audio_path" in rows[0]

            # Count positive and negative samples
            positives = [r for r in rows if r["label"] == "1"]
            negatives = [r for r in rows if r["label"] == "0"]

            # Should have some of each (given our fake data)
            assert len(positives) > 0
            assert len(negatives) > 0

    def test_audioset_create_subset_respects_positive_ratio(self, tmp_path):
        """Test that create_subset respects the positive_ratio parameter."""
        config = AudiosetConfig()

        # Mock with more data to test ratio
        config._mid_to_name = {"/m/dog": "Dog", "/m/cat": "Cat"}
        config._name_to_mid = {"Dog": 0, "Cat": 1}  # Integer ID mapping

        fake_metadata = []
        # Create 20 dog samples and 20 cat samples
        for i in range(20):
            fake_metadata.append(
                {
                    "filename": f"dog{i}.flac",
                    "labels": ["Dog"],
                    "split": "unbal_train",
                    "audio_path": Path(f"/fake/dog{i}.flac"),
                }
            )
            fake_metadata.append(
                {
                    "filename": f"cat{i}.flac",
                    "labels": ["Cat"],
                    "split": "unbal_train",
                    "audio_path": Path(f"/fake/cat{i}.flac"),
                }
            )

        with patch.object(config, "load_metadata", return_value=fake_metadata):
            output_path = tmp_path / "test_ratio.csv"
            config.create_subset(
                output_path=output_path,
                class_name="Dog",
                max_samples=10,
                positive_ratio=0.3,  # 30% positive
                seed=42,
            )

            # Read and count
            import csv

            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            positives = len([r for r in rows if r["label"] == "1"])
            total = len(rows)

            # Should be close to 30% positive (3 out of 10)
            assert total == 10
            assert positives == 3  # 30% of 10

    def test_audioset_create_subset_invalid_class_raises_error(self, tmp_path):
        """Test that create_subset raises error for invalid class name."""
        config = AudiosetConfig()

        # Mock ontology
        config._mid_to_name = {"/m/dog": "Dog"}
        config._name_to_mid = {"Dog": 0}  # Integer ID mapping

        with pytest.raises(ValueError, match="Class 'InvalidClass' not found"):
            config.create_subset(
                output_path=tmp_path / "test.csv", class_name="InvalidClass", max_samples=100
            )

    def test_audioset_create_subset_creates_parent_directory(self, tmp_path):
        """Test that create_subset creates parent directories if needed."""
        config = AudiosetConfig()

        config._mid_to_name = {"/m/dog": "Dog"}
        config._name_to_mid = {"Dog": 0}  # Integer ID mapping

        fake_metadata = [
            {
                "filename": "dog1.flac",
                "labels": ["Dog"],
                "split": "unbal_train",
                "audio_path": Path("/fake/dog1.flac"),
            },
        ]

        with patch.object(config, "load_metadata", return_value=fake_metadata):
            # Use nested path that doesn't exist
            output_path = tmp_path / "nested" / "dir" / "subset.csv"

            config.create_subset(output_path=output_path, class_name="Dog", max_samples=10, seed=42)

            # Parent directories should be created
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_audioset_create_subset_reproducible_with_seed(self, tmp_path):
        """Test that create_subset produces same results with same seed."""
        config = AudiosetConfig()

        config._mid_to_name = {"/m/dog": "Dog", "/m/cat": "Cat"}
        config._name_to_mid = {"Dog": 0, "Cat": 1}  # Integer ID mapping

        # Create larger dataset to test sampling
        fake_metadata = []
        for i in range(50):
            fake_metadata.append(
                {
                    "filename": f"dog{i}.flac",
                    "labels": ["Dog"],
                    "split": "unbal_train",
                    "audio_path": Path(f"/fake/dog{i}.flac"),
                }
            )
            fake_metadata.append(
                {
                    "filename": f"cat{i}.flac",
                    "labels": ["Cat"],
                    "split": "unbal_train",
                    "audio_path": Path(f"/fake/cat{i}.flac"),
                }
            )

        with patch.object(config, "load_metadata", return_value=fake_metadata):
            # Create two subsets with same seed
            output1 = tmp_path / "subset1.csv"
            output2 = tmp_path / "subset2.csv"

            config.create_subset(output1, "Dog", 20, seed=42)
            config.create_subset(output2, "Dog", 20, seed=42)

            # Read both files
            import csv

            with open(output1) as f:
                rows1 = list(csv.DictReader(f))
            with open(output2) as f:
                rows2 = list(csv.DictReader(f))

            # Should have identical contents
            assert len(rows1) == len(rows2)
            for r1, r2 in zip(rows1, rows2, strict=False):
                assert r1["filename"] == r2["filename"]
                assert r1["label"] == r2["label"]
