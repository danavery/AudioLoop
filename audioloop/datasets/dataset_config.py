"""
Base dataset configuration interface for AudioLoop.

Provides a common interface for all datasets used in active learning,
allowing new datasets to be added without modifying existing code.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class DatasetConfig(ABC):
    """Common interface for dataset configurations used in active learning."""

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
    def parse_metadata_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format.

        Args:
            row: Raw CSV row as dict

        Returns:
            Standardized metadata dict with consistent keys
        """
        pass

    @abstractmethod
    def get_audio_path(self, filename: str, fold: int | None = None) -> Path:
        """Get full path to audio file.

        Args:
            filename: Audio filename from metadata
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

    # === Audio Processing ===
    @abstractmethod
    def get_audio_processing_params(self) -> dict[str, Any]:
        """Get audio processing parameters for spectrogram generation.

        Returns:
            Dict with keys: sample_rate, n_fft, hop_length, n_mels, top_db, fixed_length
        """
        pass

    @abstractmethod
    def create_spectrogram_transform(self) -> nn.Sequential:
        """Create PyTorch transform pipeline for generating spectrograms.

        Returns:
            PyTorch Sequential transform for audio -> spectrogram conversion
        """
        pass

    @abstractmethod
    def get_output_shape(self) -> tuple[int, ...]:
        """Get the shape of tensors produced by this dataset.

        Returns:
            Tuple representing tensor shape (excluding batch dimension)
        """
        pass

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

    # === Audio Processing Pipeline ===
    @abstractmethod
    def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Fix spectrogram to target length by padding or cropping.

        Args:
            spec: Input spectrogram tensor

        Returns:
            Spectrogram tensor with fixed length matching dataset configuration
        """
        pass

    @abstractmethod
    def process_single_file(self, file_info: dict, output_dir: Path) -> tuple[bool, int | None]:
        """Process a single audio file and save its spectrogram.

        Args:
            file_info: Metadata dict with 'filename', 'audio_path', etc.
            output_dir: Directory to save the processed spectrogram

        Returns:
            Tuple of (success: bool, original_length: int | None)
        """
        pass
