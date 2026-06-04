"""
Base dataset configuration interface for AudioLoop.

Provides a common interface for all datasets used in active learning,
allowing new datasets to be added without modifying existing code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DatasetConfig(ABC):
    """Common interface for dataset configurations used in active learning.

    Class Attributes:
        description: Human-readable description of the dataset (optional, used for
            --list-datasets)
    """

    # === Bad Files Exclusion ===
    def get_bad_files(self) -> set[str]:
        """Return set of filenames known to crash during processing.

        Override in subclass to exclude corrupt/problematic files.
        Filenames should be base names only (e.g., 'Z3YaJ9Vi4lY.flac').
        """
        return set()

    @property
    def min_audio_file_size(self) -> int | None:
        """Minimum audio file size in bytes (files smaller are considered corrupt).

        Override in subclass to filter out suspiciously small files.
        Default: None (no size filtering).

        Example: AudioSet 10-second clips should be ~80KB+, so min_audio_file_size=9000
        catches corrupt FLAC files with headers but no data (typically 8288 bytes).
        """
        return None

    # === Core Dataset Properties ===
    @property
    @abstractmethod
    def dataset_csv(self) -> Path:
        """Path to the main dataset CSV file."""
        pass

    @property
    @abstractmethod
    def audio_root(self) -> Path:
        """Root directory containing audio files."""
        pass

    @property
    @abstractmethod
    def audio_extension(self) -> str:
        """Audio file extension for this dataset (e.g., '.wav', '.flac')."""
        pass

    @property
    @abstractmethod
    def name_to_id(self) -> dict[str, int]:
        """Mapping from class names to class IDs."""
        pass

    @property
    @abstractmethod
    def vocabulary(self) -> dict[int, str]:
        """Mapping from class IDs to class names."""
        pass

    @abstractmethod
    def list_classes(self) -> None:
        """Print available classes for this dataset."""
        pass

    # === Split Management ===
    @abstractmethod
    def get_available_splits(self) -> list[str]:
        """Return list of valid split names for this dataset.

        Returns:
            List of split names that can be passed to load_metadata()
        """
        pass

    @abstractmethod
    def get_default_split(self) -> str:
        """Return the default split name for this dataset.

        Returns:
            Default split name used when load_metadata() called without arguments
        """
        pass

    # === Metadata and File Management ===
    def load_metadata(self, split: str | None = None) -> list[dict[str, Any]]:
        """Load metadata entries for the specified split.

        Args:
            split: Dataset split to load (use get_available_splits() to see options)
                   If None, uses get_default_split()

        Returns:
            List of metadata dictionaries

        Raises:
            ValueError: If split is not in get_available_splits()
        """
        if split is None:
            split = self.get_default_split()

        available = self.get_available_splits()
        if split not in available:
            raise ValueError(
                f"Split '{split}' not available for {self.__class__.__name__}. "
                f"Valid splits: {available}"
            )

        return self._load_metadata_for_split(split)

    @abstractmethod
    def _load_metadata_for_split(self, split: str) -> list[dict[str, Any]]:
        """Internal method to load metadata for a validated split.

        Args:
            split: Validated split name from get_available_splits()

        Returns:
            List of metadata dictionaries
        """
        pass

    @abstractmethod
    def parse_metadata_row(self, row: dict[str, str], split: str | None = None) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format.

        Args:
            row: Raw CSV row as dict
            split: Optional split information (some datasets need this for path resolution)

        Returns:
            Standardized metadata dict with consistent keys
        """
        pass

    @abstractmethod
    def get_audio_path(
        self, filename: str, split: str | None = None, fold: int | None = None
    ) -> Path:
        """Get full path to audio file.

        Args:
            filename: Audio filename from metadata
            split: Dataset split (used by datasets with split-based directories like AudioSet)
            fold: Fold number (used by some datasets like UrbanSound8K)

        Returns:
            Full path to the audio file
        """
        pass

    @abstractmethod
    def get_spectrogram_path(self, filename: str, specs_dir: Path) -> Path:
        """Get path where spectrogram should be stored.

        Args:
            filename: Original audio filename
            specs_dir: Root directory for spectrograms

        Returns:
            Full path to spectrogram file
        """
        pass

    # === Feature Extraction ===
    @property
    def feature_extractor(self):
        """Lazily build and cache this dataset's feature extractor.

        The extractor owns audio->tensor production (load -> transform -> fix) AND the
        audio-processing parameters (sample_rate, n_fft, ... — feature concerns, not
        dataset identity). The dataset config is passed in only for file-level knowledge
        the build step needs (get_spectrogram_path, get_bad_files, min_audio_file_size).
        """
        fx = getattr(self, "_feature_extractor", None)
        if fx is None:
            # Local import avoids a module-level datasets <-> feature_extractor cycle.
            from audioloop.feature_extractor import SpectrogramExtractor

            fx = SpectrogramExtractor(self)
            self._feature_extractor = fx
        return fx

    # === Binary Classification ===
    @abstractmethod
    def is_positive_class(self, class_name: str, positive_class: str | int) -> bool:
        """Determine if a class matches the positive class for binary classification.

        Args:
            class_name: Class name from metadata entry
            positive_class: Target positive class (name or ID)

        Returns:
            True if this is a positive class, False otherwise
        """
        pass

    @abstractmethod
    def get_binary_label(
        self, item: dict[str, Any], positive_class_id: int, positive_class_name: str
    ) -> int:
        """Get binary label for an item based on positive class criteria.

        Args:
            item: Metadata item from parse_metadata_row()
            positive_class_id: ID of the positive class
            positive_class_name: Name of the positive class

        Returns:
            1 if positive class, 0 otherwise
        """
        pass

    # === Dataset Subsetting ===
    def create_subset(
        self,
        output_path: Path,
        class_name: str,
        max_samples: int,
        positive_ratio: float = 0.5,
        split: str | None = None,
        seed: int = 42,
    ) -> Path:
        """Create a training-ready subset CSV for binary classification.

        Creates a CSV file with columns: filename, label, original_class, split, audio_path
        This CSV can be used directly for training and supports lazy spectrogram generation.

        Args:
            output_path: Where to write the subset CSV
            class_name: Class name to use as positive class for binary classification
            max_samples: Maximum total samples (positive + negative combined)
            positive_ratio: Target ratio of positive samples (0.0-1.0, default 0.5)
            split: Dataset split to sample from (unified split/fold parameter):
                   - FSD50K: "dev" or "eval"
                   - UrbanSound8K: "all", "fold1", ..., "fold10"
                   - AudioSet: "bal_train", "unbal_train", "eval"
                   If None, uses get_default_split()
            seed: Random seed for reproducibility (default 42)

        Returns:
            Path to created CSV file (same as output_path)

        Raises:
            ValueError: If class_name not found in dataset vocabulary
            ValueError: If split not in get_available_splits()
            NotImplementedError: If dataset doesn't support subsetting yet
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support create_subset(). "
            "Dataset subsetting may not be necessary for datasets that are small enough "
            "to use in their entirety."
        )

    # === Custom Metadata Support ===
    def supports_custom_csv(self) -> bool:
        """Return True if this dataset supports custom CSV files/subsets.

        Override in subclasses that support custom metadata sources.

        Returns:
            False by default - datasets must opt-in to custom support
        """
        return False

    def set_custom_csv(self, custom_csv_path: Path) -> None:
        """Set custom CSV file to use instead of default dataset metadata.

        Args:
            custom_csv_path: Path to custom CSV file

        Raises:
            NotImplementedError: If dataset doesn't support custom CSV files
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't support custom CSV files")
