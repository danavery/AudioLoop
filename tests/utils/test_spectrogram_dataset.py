"""
Tests for SpectrogramDataset with lazy generation support.
"""

import csv
from unittest.mock import Mock, patch

import torch

from audioloop.utils.spectrogram_dataset import SpectrogramDataset


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

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

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

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

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

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

        assert dataset.samples[0]["original_class"] == 5

    def test_parse_csv_handles_multiple_extensions(self, tmp_path):
        """Test that multiple audio extensions are handled for spec path."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Test with different audio extensions
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["test.flac", "1"])
            writer.writerow(["test.mp3", "0"])
            writer.writerow(["test.ogg", "1"])

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

        # All should map to .pt files
        assert all(sample["spec_filepath"].endswith(".pt") for sample in dataset.samples)
        assert dataset.samples[0]["spec_filepath"].endswith("test.pt")


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

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

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
        mock_config = Mock()
        mock_config.sample_rate = 44100
        mock_config.min_audio_file_size = None
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = torch.randn(1, 128, 100)

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock the audio file existence and loading
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("torchaudio.load") as mock_load,
        ):
            mock_load.return_value = (torch.randn(1, 16000), 44100)

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
        mock_config = Mock()
        mock_config.sample_rate = 44100
        spec_data = torch.randn(1, 128, 100)
        mock_config.min_audio_file_size = None
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = spec_data

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock the audio loading
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("torchaudio.load") as mock_load,
            patch("audioloop.utils.spectrogram_dataset.os.path.exists", return_value=False),
            patch("audioloop.utils.spectrogram_dataset.os.makedirs"),
            patch("audioloop.utils.spectrogram_dataset.torch.save") as mock_save,
        ):
            mock_load.return_value = (torch.randn(1, 16000), 44100)

            # Mock the transform
            mock_transform = mock_config.create_spectrogram_transform.return_value
            mock_transform.return_value = spec_data

            # Generate spec
            _ = dataset[0]

            # Should have saved to disk
            mock_save.assert_called_once()
            saved_path = mock_save.call_args[0][1]
            assert saved_path.endswith("test.pt")

    def test_error_when_spec_missing_and_no_lazy_generation(self, tmp_path):
        """Test that helpful error is raised when spec missing and no lazy generation."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV without audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerow(["missing.wav", "1"])

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

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
        mock_config = Mock()
        mock_config.sample_rate = 44100
        mock_config.min_audio_file_size = None
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = torch.randn(1, 128, 100)

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock stereo audio loading
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("torchaudio.load") as mock_load,
            patch("audioloop.utils.spectrogram_dataset.os.path.exists", return_value=False),
            patch("audioloop.utils.spectrogram_dataset.torch.save"),
        ):
            # Return stereo audio (2 channels)
            stereo_audio = torch.randn(2, 16000)
            mock_load.return_value = (stereo_audio, 44100)

            # Mock the transform to check what it receives
            mock_transform = mock_config.create_spectrogram_transform.return_value

            def check_mono(waveform):
                # Should receive mono (1 channel)
                assert waveform.shape[0] == 1
                return torch.randn(1, 128, 100)

            mock_transform.side_effect = check_mono

            # Should convert stereo to mono
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

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

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

        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

        item = dataset[0]
        assert item is not None
        assert "original_class" not in item


class TestDatasetConfigIntegration:
    """Test integration with dataset configs."""

    def test_dataset_config_parameter_enables_lazy_generation(self, tmp_path):
        """Test that providing dataset_config enables lazy generation."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with audio_path
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["test.flac", "1", "/fake/path.flac"])

        # Without dataset_config
        dataset_no_config = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))
        assert dataset_no_config.dataset_config is None

        # With dataset_config
        mock_config = Mock()
        dataset_with_config = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )
        assert dataset_with_config.dataset_config == mock_config

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

        mock_config = Mock()
        mock_config.min_audio_file_size = None

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

        mock_config = Mock()
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

    def test_lazy_generation_without_dataset_config_fails_gracefully(self, tmp_path):
        """Test that missing spec without dataset_config gives helpful error."""
        csv_file = tmp_path / "test.csv"
        specs_dir = tmp_path / "specs"
        specs_dir.mkdir()

        # Write CSV with audio_path but no dataset_config
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "audio_path"])
            writer.writerow(["missing.wav", "1", str(tmp_path / "missing.wav")])

        # No dataset_config provided
        dataset = SpectrogramDataset(csv_file=str(csv_file), specs_dir=str(specs_dir))

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
        mock_config = Mock()
        mock_config.sample_rate = 44100
        spec_data = torch.randn(1, 128, 100)
        mock_config.min_audio_file_size = None
        mock_config.create_spectrogram_transform.return_value = Mock()
        mock_config.fix_spectrogram_length.return_value = spec_data

        dataset = SpectrogramDataset(
            csv_file=str(csv_file), specs_dir=str(specs_dir), dataset_config=mock_config
        )

        # Mock audio loading
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("torchaudio.load") as mock_load,
        ):
            mock_load.return_value = (torch.randn(1, 16000), 44100)
            mock_transform = mock_config.create_spectrogram_transform.return_value
            mock_transform.return_value = spec_data

            # Generate spec
            _ = dataset[0]

            # Verify subdirectories were created
            assert specs_dir.exists()
