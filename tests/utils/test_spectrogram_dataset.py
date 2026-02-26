"""
Tests for SpectrogramDataset with lazy generation support.
"""

import csv
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from audioloop.utils.spectrogram_dataset import SpectrogramDataset


def _make_mock_config():
    """Create a mock dataset config with get_spectrogram_path that mimics standard behavior."""
    mock_config = Mock()
    mock_config.sample_rate = 44100
    mock_config.min_audio_file_size = None

    def _get_spec_path(filename, specs_dir):
        """Replace common audio extensions with .pt"""
        spec_name = filename
        for ext in [".wav", ".flac", ".mp3", ".ogg"]:
            spec_name = spec_name.replace(ext, ".pt")
        return Path(specs_dir) / spec_name

    mock_config.get_spectrogram_path = _get_spec_path
    return mock_config


class TestSpectrogramDatasetCSVParsing:
    """Test CSV parsing with different formats."""

    def test_parse_csv_with_audio_path_column(self, tmp_path):
        """Test that CSV with audio_path column is parsed correctly."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with audio_path column
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class", "split", "audio_path"])
            writer.writerow(["test.flac", "1", "Dog", "bal_train", "/mnt/audio/test.flac"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        assert len(dataset.samples) == 1
        assert dataset.samples[0]["audio_path"] == "/mnt/audio/test.flac"
        assert dataset.samples[0]["label"] == 1
        assert dataset.samples[0]["original_class"] == "Dog"

    def test_parse_csv_with_string_original_class(self, tmp_path):
        """Test that original_class as string (class name) is handled."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with string original_class
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class"])
            writer.writerow(["test.wav", "1", "Speech"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        assert dataset.samples[0]["original_class"] == "Speech"

    def test_parse_csv_with_int_original_class(self, tmp_path):
        """Test that original_class as int (class ID) is handled."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with int original_class
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class"])
            writer.writerow(["test.wav", "1", "5"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        assert dataset.samples[0]["original_class"] == 5

    def test_spec_path_uses_dataset_config(self, tmp_path):
        """Test that spec path is resolved via dataset_config.get_spectrogram_path."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.flac", "1"])
            writer.writerow(["test.mp3", "0"])
            writer.writerow(["test.ogg", "1"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        # All should map to .pt files
        assert all(sample["spec_filepath"].endswith(".pt") for sample in dataset.samples)
        assert dataset.samples[0]["spec_filepath"].endswith("test.pt")

    def test_dataset_config_required(self, tmp_path):
        """Test that omitting dataset_config raises ValueError."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.wav", "1"])

        import pytest

        with pytest.raises(ValueError, match="dataset_config is required"):
            SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))


class TestLazySpectrogramGeneration:
    """Test lazy spectrogram generation functionality."""

    def test_load_existing_spec_file(self, tmp_path):
        """Test that existing spec files are loaded directly."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Create a fake spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = specs_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.wav", "1"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        # Should load the existing file
        item = dataset[0]
        assert item is not None
        assert torch.allclose(item["data"], spec_data)

    def test_lazy_generation_when_spec_missing(self, tmp_path):
        """Test that spec is generated when missing and audio_path available."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_dir / "test.flac")])

        # Mock dataset config
        mock_config = _make_mock_config()
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = torch.randn(1, 128, 100)

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock the audio file existence and loading
        mock_config.load_audio.return_value = torch.randn(1, 16000)
        with patch("pathlib.Path.exists", return_value=True):
            # Mock the transform
            mock_transform = mock_config.create_spectrogram_transform.return_value
            mock_transform.return_value = torch.randn(1, 128, 100)

            # Should generate spec
            item = dataset[0]
            assert item is not None
            assert "data" in item
            assert item["data"].shape == (1, 128, 100)

    def test_lazy_generation_caches_to_disk(self, tmp_path):
        """Test that generated specs are cached to disk."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test.flac"

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_file)])

        # Mock dataset config
        mock_config = _make_mock_config()
        spec_data = torch.randn(1, 128, 100)
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = spec_data

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock the audio loading
        mock_config.load_audio.return_value = torch.randn(1, 16000)
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("audioloop.utils.spectrogram_dataset.os.path.exists", return_value=False),
            patch("audioloop.utils.spectrogram_dataset.os.makedirs"),
            patch("audioloop.utils.spectrogram_dataset.torch.save") as mock_save,
        ):
            # Mock the transform
            mock_transform = mock_config.create_spectrogram_transform.return_value
            mock_transform.return_value = spec_data

            # Generate spec
            _ = dataset[0]

            # Should have saved to disk
            mock_save.assert_called_once()
            saved_path = mock_save.call_args[0][1]
            assert saved_path.endswith("test.pt")

    def test_error_when_spec_missing_and_no_audio_path(self, tmp_path):
        """Test that helpful error is raised when spec missing and no audio_path."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV without audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["missing.wav", "1"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        # Should return None for missing files (graceful skip behavior)
        result = dataset[0]
        assert result is None

    def test_stereo_to_mono_conversion(self, tmp_path):
        """Test that stereo audio is converted to mono."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_dir / "test.flac")])

        # Mock dataset config
        mock_config = _make_mock_config()
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = torch.randn(1, 128, 100)

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock audio loading - load_audio returns mono since torchcodec handles conversion
        mock_config.load_audio.return_value = torch.randn(1, 16000)
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("audioloop.utils.spectrogram_dataset.os.path.exists", return_value=False),
            patch("audioloop.utils.spectrogram_dataset.torch.save"),
        ):
            # Mock the transform to check what it receives
            mock_transform = mock_config.create_spectrogram_transform.return_value

            def check_mono(waveform):
                # Should receive mono (1 channel) from load_audio
                assert waveform.shape[0] == 1
                return torch.randn(1, 128, 100)

            mock_transform.side_effect = check_mono

            # Should receive mono audio from load_audio
            _ = dataset[0]


class TestSpectrogramDatasetReturnValues:
    """Test the structure of returned items."""

    def test_return_dict_includes_all_fields(self, tmp_path):
        """Test that returned dict includes all expected fields."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Create a spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = specs_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV with all optional fields
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class"])
            writer.writerow(["test.wav", "1", "Dog"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
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
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Create a spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = specs_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV without original_class
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.wav", "1"])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        item = dataset[0]
        assert item is not None
        assert "original_class" not in item


class TestDatasetConfigIntegration:
    """Test integration with dataset configs."""

    def test_dataset_config_stored(self, tmp_path):
        """Test that dataset_config is stored on the instance."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", "/fake/path.flac"])

        mock_config = _make_mock_config()
        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )
        assert dataset.dataset_config == mock_config

    def test_error_when_audio_file_missing(self, tmp_path):
        """Test that FileNotFoundError is raised when audio file doesn't exist."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with non-existent audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", "/nonexistent/path.flac"])

        mock_config = _make_mock_config()

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        with patch("audioloop.utils.spectrogram_dataset.os.path.exists", return_value=False):
            # Should return None for missing audio file (graceful skip behavior)
            result = dataset[0]
            assert result is None

    def test_lazy_generation_skipped_when_spec_exists(self, tmp_path):
        """Test that lazy generation is skipped when spec file already exists."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Create existing spec file
        spec_data = torch.randn(1, 128, 100)
        spec_file = specs_dir / "test.pt"
        torch.save(spec_data, spec_file)

        # Write CSV with audio_path (but spec already exists)
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.wav", "1", str(tmp_path / "test.wav")])

        mock_config = _make_mock_config()
        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Load the item - should use existing file, not trigger lazy generation
        item = dataset[0]
        assert item is not None
        # Should have loaded the existing spec
        assert torch.allclose(item["data"], spec_data)

        # Mock config methods should not have been called
        mock_config.create_spectrogram_transform.assert_not_called()

    def test_lazy_generation_without_audio_path_fails_gracefully(self, tmp_path):
        """Test that missing spec without audio_path gives graceful skip."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with audio_path but no dataset_config
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["missing.wav", "1", str(tmp_path / "missing.wav")])

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=_make_mock_config()
        )

        # Should return None for files that can't be loaded (graceful skip behavior)
        result = dataset[0]
        assert result is None

    def test_lazy_generation_creates_subdirectories(self, tmp_path):
        """Test that lazy generation creates necessary subdirectories for specs."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs" / "subdir"  # Nested directory

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        audio_file = audio_dir / "test.flac"

        # Write CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", str(audio_file)])

        # Mock dataset config
        mock_config = _make_mock_config()
        spec_data = torch.randn(1, 128, 100)
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = spec_data

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock audio loading
        mock_config.load_audio.return_value = torch.randn(1, 16000)
        with patch("pathlib.Path.exists", return_value=True):
            mock_transform = mock_config.create_spectrogram_transform.return_value
            mock_transform.return_value = spec_data

            # Generate spec
            _ = dataset[0]

            # Verify subdirectories were created
            assert specs_dir.exists()
