import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn

from audioloop.utils.log_normalize import LogNormalize
from audioloop.utils.paths import get_project_root

from .dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


# UrbanSound8K class mapping
URBANSOUND8K_CLASSES = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music",
}

# Reverse mapping for convenience
CLASS_NAME_TO_ID = {name: class_id for class_id, name in URBANSOUND8K_CLASSES.items()}


def load_urbansound8k_vocabulary() -> dict[int, str]:
    """Load UrbanSound8K vocabulary mapping.

    Returns hardcoded mapping since UrbanSound8K doesn't provide a vocabulary file.

    Returns:
        Dictionary mapping class_id to class_name
    """
    return URBANSOUND8K_CLASSES.copy()


def get_class_name(class_id: int, vocab_path: Path | None = None) -> str:
    """Get human-readable class name from UrbanSound8K class ID.

    Args:
        class_id: UrbanSound8K class ID (0-9)
        vocab_path: Ignored for UrbanSound8K (kept for interface consistency)

    Returns:
        Human-readable class name

    Raises:
        ValueError: If class_id is not valid
    """
    vocabulary = load_urbansound8k_vocabulary()
    if class_id not in vocabulary:
        raise ValueError(f"Invalid class ID: {class_id}. Must be 0-9.")
    return vocabulary[class_id]


def get_class_id(class_name: str, vocab_path: Path | None = None) -> int:
    """Get UrbanSound8K class ID from human-readable class name.

    Args:
        class_name: Human-readable class name
        vocab_path: Ignored for UrbanSound8K (kept for interface consistency)

    Returns:
        UrbanSound8K class ID (0-9)

    Raises:
        ValueError: If class_name is not valid
    """
    vocabulary = load_urbansound8k_vocabulary()
    name_to_id = {name: class_id for class_id, name in vocabulary.items()}

    if class_name not in name_to_id:
        valid_names = list(name_to_id.keys())
        raise ValueError(f"Invalid class name: '{class_name}'. Valid names: {valid_names}")
    return name_to_id[class_name]


def list_classes(vocab_path: Path | None = None) -> None:
    """Print all available UrbanSound8K classes.

    Args:
        vocab_path: Ignored for UrbanSound8K (kept for interface consistency)
    """
    vocabulary = load_urbansound8k_vocabulary()
    print("UrbanSound8K Classes:")
    print("=" * 30)
    for class_id in sorted(vocabulary.keys()):
        name = vocabulary[class_id]
        print(f"{class_id}: {name}")


@dataclass
class UrbanSound8KConfig(DatasetConfig):
    """Configuration for UrbanSound8K dataset."""

    # Audio processing parameters
    _sample_rate: int = 44100
    _n_fft: int = 1024
    _hop_length: int = 256
    _n_mels: int = 128
    _top_db: int = 80
    _max_spectrogram_length: int = 993  # Max length from training data analysis

    # Default paths
    metadata_csv: Path = Path("data/urbansound8k/UrbanSound8K.csv")
    _dataset_csv: Path = Path("data/urbansound8k/UrbanSound8K.csv")  # Common interface
    _audio_root: Path = Path("data/urbansound8k")


    # === Core Dataset Properties ===
    @property
    def dataset_csv(self) -> Path:
        """Path to the main dataset CSV file."""
        return self._dataset_csv

    @property
    def audio_root(self) -> Path:
        """Root directory containing audio files."""
        if self._audio_root.is_absolute():
            return self._audio_root
        # Resolve relative paths against project root
        return get_project_root() / self._audio_root

    @property
    def vocabulary(self) -> dict[int, str]:
        """Get UrbanSound8K vocabulary mapping."""
        return load_urbansound8k_vocabulary()

    @property
    def name_to_id(self) -> dict[str, int]:
        """Get class name to ID mapping."""
        return {name: class_id for class_id, name in self.vocabulary.items()}

    @property
    def audio_extension(self) -> str:
        """Audio file extension for UrbanSound8K (.wav)."""
        return ".wav"

    def list_classes(self) -> None:
        """Print all available UrbanSound8K classes."""
        print("UrbanSound8K Classes:")
        print("=" * 30)
        for class_id in sorted(self.vocabulary.keys()):
            name = self.vocabulary[class_id]
            print(f"{class_id}: {name}")

    # === Metadata and File Management ===

    # === Split Management ===
    def get_available_splits(self) -> list[str]:
        """Return list of valid split names for UrbanSound8K dataset."""
        return ["all"]

    def get_default_split(self) -> str:
        """Return the default split name for UrbanSound8K dataset."""
        return "all"

    def _load_metadata_for_split(self, split: str) -> list[dict]:
        """Load all metadata for the UrbanSound8K dataset."""
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_csv}")

        audio_files = []
        with self.metadata_csv.open("r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                parsed = self.parse_metadata_row(row)
                audio_files.append(parsed)

        return audio_files

    def parse_metadata_row(self, row: dict[str, str], split: str | None = None) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format."""
        class_name = row["class"]
        return {
            "filename": row["slice_file_name"],
            "class_id": int(row["classID"]),
            "class_name": class_name,
            "labels": [class_name],  # Convert single label to array for consistency
            "fold": int(row["fold"]),
            "audio_path": self.get_audio_path(row["slice_file_name"], fold=int(row["fold"])),
        }

    def get_audio_path(
        self, filename: str, split: str | None = None, fold: int | None = None
    ) -> Path:
        """Get full path to audio file.

        Args:
            filename: Audio filename from metadata
            split: Ignored for UrbanSound8K (uses fold structure instead)
            fold: Fold number (1-10) - determines subdirectory

        Returns:
            Full path to audio file in fold{fold}/ subdirectory
        """
        if fold is not None:
            return self.audio_root / f"fold{fold}" / filename
        # Try to find the file in any fold (less efficient but works)
        for fold_num in range(1, 11):
            path = self.audio_root / f"fold{fold_num}" / filename
            if path.exists():
                return path
        # Return most likely path if not found
        return self.audio_root / "fold1" / filename

    def get_spectrogram_path(self, filename: str, specs_dir: Path) -> Path:
        """Get path where spectrogram should be stored."""
        spec_filename = filename.replace(".wav", ".pt")
        return specs_dir / spec_filename

    # === Audio Processing ===

    def get_audio_processing_params(self) -> dict[str, Any]:
        """Get audio processing parameters for spectrogram generation."""
        return {
            "sample_rate": self._sample_rate,
            "n_fft": self._n_fft,
            "hop_length": self._hop_length,
            "n_mels": self._n_mels,
            "top_db": self._top_db,
            "max_spectrogram_length": self._max_spectrogram_length,
        }

    def create_spectrogram_transform(self):
        """Create PyTorch transform pipeline for generating spectrograms."""
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                n_fft=self._n_fft,
                hop_length=self._hop_length,
                n_mels=self._n_mels,
            ),
            LogNormalize(top_db=self._top_db),
        )

    def get_output_shape(self) -> tuple[int, ...]:
        """Get the shape of tensors produced by this dataset."""
        return (self._n_mels, -1)  # -1 indicates variable time dimension

    def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Fix spectrogram length by cropping outliers but preserving natural variation."""
        current_length = spec.shape[-1]  # Time dimension is last
        max_length = self._max_spectrogram_length

        # Only crop if it exceeds reasonable maximum (handles outliers)
        if current_length > max_length:
            # Crop from the center
            start_idx = (current_length - max_length) // 2
            spec = spec[..., start_idx : start_idx + max_length]

        # Don't pad short spectrograms - preserve natural length
        return spec

    def process_single_file(self, file_info: dict, output_dir: Path) -> tuple[bool, int | None]:
        """Process a single audio file and save its spectrogram."""
        try:
            audio_path = file_info["audio_path"]
            filename = file_info["filename"]

            # Check if audio file exists
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return False, None

            # Load audio (resamples and converts to mono)
            waveform = self.load_audio(audio_path)

            # Create spectrogram
            spec_transform = self.create_spectrogram_transform()
            spec = spec_transform(waveform)

            # Store original length before fixing
            original_length = spec.shape[-1]

            # Fix spectrogram length
            spec = self.fix_spectrogram_length(spec)

            # Save spectrogram
            output_filename = filename.replace(".wav", ".pt")
            output_path = output_dir / output_filename
            torch.save(spec, output_path)

            return True, original_length

        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            return False, None

    # === Binary Classification ===
    def is_positive_class(self, class_name: str, positive_class: str | int) -> bool:
        """Determine if a class matches the positive class for binary classification."""
        if isinstance(positive_class, str):
            return class_name == positive_class
        if isinstance(positive_class, int):
            return (
                positive_class in self.vocabulary and self.vocabulary[positive_class] == class_name
            )
        return False

    def get_binary_label(
        self, item: dict[str, Any], positive_class_id: int, positive_class_name: str
    ) -> int:
        """Get binary label for an item based on positive class criteria."""
        return 1 if item["class_id"] == positive_class_id else 0
