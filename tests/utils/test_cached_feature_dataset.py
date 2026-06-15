"""
Tests for CachedFeatureDataset with lazy generation support.
"""

import csv
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from audioloop.utils.cached_feature_dataset import CachedFeatureDataset


def _make_mock_extractor():
    """Create a mock SpectrogramExtractor wrapping a mock dataset config.

    CachedFeatureDataset uses the extractor for lazy generation (extract_one) and cached-feature
    path resolution (get_cached_feature_path), and reaches its dataset_config only for the
    corruption guard (min_audio_file_size). Tests stub the seam:
    extractor.extract_one.return_value = [<spec tensor>] (a list of segments; the lazy
    path is single-segment). The audio->tensor composition itself is covered by
    tests/test_feature_extractor.py.
    """
    mock_config = Mock()
    mock_config.min_audio_file_size = None

    def _get_spec_path(filename, feature_cache_dir):
        """Replace common audio extensions with .pt"""
        spec_name = filename
        for ext in [".wav", ".flac", ".mp3", ".ogg"]:
            spec_name = spec_name.replace(ext, ".pt")
        return Path(feature_cache_dir) / spec_name

    extractor = Mock()
    extractor.dataset_config = mock_config
    extractor.get_cached_feature_path = _get_spec_path
    return extractor


class TestCachedFeatureDatasetCSVParsing:
    """Test CSV parsing with different formats."""

    def test_parse_csv_with_audio_path_column(self, tmp_path):
        """Test that CSV with audio_path column is parsed correctly."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV with audio_path column
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class", "split", "audio_path"])
            writer.writerow(["test.flac", "1", "Dog", "bal_train", "/mnt/audio/test.flac"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        assert len(dataset.samples) == 1
        assert dataset.samples[0]["audio_path"] == "/mnt/audio/test.flac"
        assert dataset.samples[0]["label"] == 1
        assert dataset.samples[0]["original_class"] == "Dog"

    def test_parse_csv_with_string_original_class(self, tmp_path):
        """Test that original_class as string (class name) is handled."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV with string original_class
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class"])
            writer.writerow(["test.wav", "1", "Speech"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        assert dataset.samples[0]["original_class"] == "Speech"

    def test_parse_csv_with_int_original_class(self, tmp_path):
        """Test that original_class as int (class ID) is handled."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV with int original_class
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class"])
            writer.writerow(["test.wav", "1", "5"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        assert dataset.samples[0]["original_class"] == 5

    def test_spec_path_uses_extractor(self, tmp_path):
        """Test that spec path is resolved via extractor.get_cached_feature_path."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.flac", "1"])
            writer.writerow(["test.mp3", "0"])
            writer.writerow(["test.ogg", "1"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        # All should map to .pt files
        assert all(sample["spec_filepath"].endswith(".pt") for sample in dataset.samples)
        assert dataset.samples[0]["spec_filepath"].endswith("test.pt")

    def test_extractor_required(self, tmp_path):
        """Test that omitting the extractor raises ValueError."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.wav", "1"])

        with pytest.raises(ValueError, match="extractor is required"):
            CachedFeatureDataset(csv_file=str(csv_file), extractor=None, feature_cache_dir=str(feature_cache_dir))


class TestLazySpectrogramGeneration:
    """Test lazy spectrogram generation functionality."""

    def test_load_existing_spec_file(self, tmp_path):
        """Test that existing spec files are loaded directly."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Create a fake spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = feature_cache_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.wav", "1"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        # Should load the existing file
        item = dataset[0]
        assert item is not None
        assert torch.allclose(item["data"], spec_data)

    def test_lazy_generation_when_spec_missing(self, tmp_path):
        """Test that spec is generated when missing and audio_path available."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test.flac"
        audio_file.write_bytes(b"audio")  # real file so audio_path.exists() is genuinely true

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_file)])

        # Mock dataset config: the extractor produces the spec.
        extractor = _make_mock_extractor()
        extractor.extract_one.return_value = [torch.randn(1, 128, 100)]

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=extractor
        )

        # Spec is missing but the audio exists, so it is generated on the fly.
        item = dataset[0]
        assert item is not None
        assert "data" in item
        assert item["data"].shape == (1, 128, 100)

    def test_lazy_generation_caches_to_disk(self, tmp_path):
        """Test that generated specs are cached to disk."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test.flac"
        audio_file.write_bytes(b"audio")  # real file so lazy generation proceeds

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_file)])

        # Mock dataset config: the extractor produces the spec.
        extractor = _make_mock_extractor()
        spec_data = torch.randn(1, 128, 100)
        extractor.extract_one.return_value = [spec_data]

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=extractor
        )

        # The cached .pt genuinely doesn't exist yet, so generation must write it; stub only
        # the write so we can assert the cache path without touching disk.
        with patch("audioloop.utils.cached_feature_dataset.torch.save") as mock_save:
            _ = dataset[0]

            # Should have saved to disk
            mock_save.assert_called_once()
            saved_path = mock_save.call_args[0][1]
            assert saved_path.endswith("test.pt")

    def test_error_when_spec_missing_and_no_audio_path(self, tmp_path):
        """Test that helpful error is raised when spec missing and no audio_path."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV without audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["missing.wav", "1"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        # Should return None for missing files (graceful skip behavior)
        result = dataset[0]
        assert result is None

    # Note: mono conversion is now the extractor's concern (SpectrogramExtractor._load_audio
    # decodes via torchcodec num_channels=1); CachedFeatureDataset no longer touches waveforms.
    # See tests/test_feature_extractor.py.


class TestCachedFeatureDatasetReturnValues:
    """Test the structure of returned items."""

    def test_return_dict_includes_all_fields(self, tmp_path):
        """Test that returned dict includes all expected fields."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Create a spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = feature_cache_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV with all optional fields
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class"])
            writer.writerow(["test.wav", "1", "Dog"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        item = dataset[0]
        assert item is not None
        assert "data" in item
        assert "label" in item
        assert "filename" in item
        assert "filepath" in item
        assert "original_class" in item
        assert item["original_class"] == "Dog"

    def test_return_dict_excludes_none_original_class(self, tmp_path):
        """Test that original_class is excluded when None."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Create a spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = feature_cache_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV without original_class
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.wav", "1"])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        item = dataset[0]
        assert item is not None
        assert "original_class" not in item


class TestDatasetConfigIntegration:
    """Test integration with dataset configs."""

    def test_extractor_stored(self, tmp_path):
        """Test that the extractor (and its dataset_config) is stored on the instance."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", "/fake/path.flac"])

        extractor = _make_mock_extractor()
        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=extractor
        )
        assert dataset.extractor is extractor
        assert dataset.dataset_config is extractor.dataset_config

    def test_error_when_audio_file_missing(self, tmp_path):
        """Test that FileNotFoundError is raised when audio file doesn't exist."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV with non-existent audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", "/nonexistent/path.flac"])

        extractor = _make_mock_extractor()

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=extractor
        )

        with patch("audioloop.utils.cached_feature_dataset.os.path.exists", return_value=False):
            # Should return None for missing audio file (graceful skip behavior)
            result = dataset[0]
            assert result is None

    def test_lazy_generation_skipped_when_spec_exists(self, tmp_path):
        """Test that lazy generation is skipped when spec file already exists."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Create existing spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = feature_cache_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV with audio_path (but spec already exists)
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.wav", "1", str(tmp_path / "test.wav")])

        extractor = _make_mock_extractor()
        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=extractor
        )

        # Load the item - should use existing file, not trigger lazy generation
        item = dataset[0]
        assert item is not None
        # Should have loaded the existing spec
        assert torch.allclose(item["data"], spec_data)

        # Extractor should not have been invoked (cached spec used directly)
        extractor.extract_one.assert_not_called()

    def test_lazy_generation_without_audio_path_fails_gracefully(self, tmp_path):
        """Test that missing spec without audio_path gives graceful skip."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs"
        feature_cache_dir.mkdir()

        # Write CSV with audio_path but no dataset_config
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["missing.wav", "1", str(tmp_path / "missing.wav")])

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=_make_mock_extractor()
        )

        # Should return None for files that can't be loaded (graceful skip behavior)
        result = dataset[0]
        assert result is None

    def test_lazy_generation_creates_subdirectories(self, tmp_path):
        """Test that lazy generation creates necessary subdirectories for specs."""
        csv_file = tmp_path / "test.csv"
        feature_cache_dir = tmp_path / "specs" / "subdir"  # Nested directory

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test.flac"
        audio_file.write_bytes(b"audio")  # real file so lazy generation proceeds

        # Write CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_file)])

        # Mock dataset config: the extractor produces the spec.
        extractor = _make_mock_extractor()
        spec_data = torch.randn(1, 128, 100)
        extractor.extract_one.return_value = [spec_data]

        dataset = CachedFeatureDataset(
            csv_file=str(csv_file), feature_cache_dir=str(feature_cache_dir), extractor=extractor
        )

        # Generate spec (nothing mocked, so caching really runs and creates the nested dir).
        _ = dataset[0]

        # Verify the nested specs directory was created on the real filesystem.
        assert feature_cache_dir.exists()
