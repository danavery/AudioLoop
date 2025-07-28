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

    # === Metadata and File Management ===
    @abstractmethod
    def load_metadata(self, split: str = "dev") -> list[dict[str, Any]]:
        """Load metadata entries for the specified split.

        Args:
            split: Dataset split to load ('dev', 'eval', etc.)

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
