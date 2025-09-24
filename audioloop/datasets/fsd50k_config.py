import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn

from audioloop.utils.log_normalize import LogNormalize

from .dataset_config import DatasetConfig

logger = logging.getLogger(__name__)


def load_fsd50k_vocabulary(vocab_path: Path) -> dict[int, str]:
    """Load FSD50K vocabulary mapping from CSV file.

    Args:
        vocab_path: Path to vocabulary.csv file

    Returns:
        Dictionary mapping class_id to class_name
    """
    vocabulary = {}
    with vocab_path.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            class_id, class_name, mid = row
            vocabulary[int(class_id)] = class_name
    return vocabulary


# Predefined semantic groups for common binary classification tasks
# These use actual FSD50K class names from the vocabulary
SEMANTIC_GROUPS = {
    "musical_instruments": {
        "Accordion",
        "Acoustic_guitar",
        "Bass_drum",
        "Bass_guitar",
        "Bowed_string_instrument",
        "Brass_instrument",
        "Crash_cymbal",
        "Cymbal",
        "Drum",
        "Drum_kit",
        "Electric_guitar",
        "Guitar",
        "Harmonica",
        "Keyboard_(musical)",
        "Music",
        "Musical_instrument",
        "Organ",
        "Piano",
        "Plucked_string_instrument",
        "Rattle_(instrument)",
        "Snare_drum",
        "Trumpet",
        "Wind_instrument_and_woodwind_instrument",
    },
    "human_sounds": {
        "Child_speech_and_kid_speaking",
        "Conversation",
        "Crowd",
        "Female_speech_and_woman_speaking",
        "Laughter",
        "Male_speech_and_man_speaking",
        "Speech",
        "Whispering",
        "Yell",
    },
    "animal_sounds": {
        "Animal",
        "Bark",
        "Bird",
        "Cat",
        "Dog",
        "Domestic_animals_and_pets",
        "Livestock_and_farm_animals_and_working_animals",
    },
    "vehicle_sounds": {
        "Accelerating_and_revving_and_vroom",
        "Aircraft",
        "Bus",
        "Car",
        "Motorcycle",
        "Train",
        "Truck",
        "Vehicle",
    },
    "mechanical_sounds": {"Drill", "Power_tool", "Tools"},
}


def get_class_name(class_id: int, vocab_path: Path | None = None) -> str:
    """Get human-readable class name from FSD50K class ID."""
    if vocab_path is None:
        vocab_path = Path("data/FSD50K/FSD50K.ground_truth/vocabulary.csv")

    vocabulary = load_fsd50k_vocabulary(vocab_path)
    if class_id not in vocabulary:
        raise ValueError(f"Invalid class ID: {class_id}. Must be 0-199.")
    return vocabulary[class_id]


def get_class_id(class_name: str, vocab_path: Path | None = None) -> int:
    """Get FSD50K class ID from human-readable class name."""
    if vocab_path is None:
        vocab_path = Path("data/FSD50K/FSD50K.ground_truth/vocabulary.csv")

    vocabulary = load_fsd50k_vocabulary(vocab_path)
    name_to_id = {name: class_id for class_id, name in vocabulary.items()}

    if class_name not in name_to_id:
        raise ValueError(
            f"Invalid class name: '{class_name}'. Use list_classes() to see valid names."
        )
    return name_to_id[class_name]


def list_classes(vocab_path: Path | None = None) -> None:
    """Print all available FSD50K classes."""
    if vocab_path is None:
        vocab_path = Path("data/FSD50K/FSD50K.ground_truth/vocabulary.csv")

    vocabulary = load_fsd50k_vocabulary(vocab_path)
    print("FSD50K Classes (200 total):")
    print("=" * 40)
    for class_id in sorted(vocabulary.keys()):
        name = vocabulary[class_id]
        print(f"{class_id:3d}: {name}")


def list_semantic_groups() -> None:
    """Print all predefined semantic groups."""
    print("FSD50K Semantic Groups:")
    print("=" * 30)
    for group_name, classes in SEMANTIC_GROUPS.items():
        print(f"\n{group_name} ({len(classes)} classes):")
        for class_name in sorted(classes):
            print(f"  - {class_name}")


@dataclass
class FSD50KConfig(DatasetConfig):
    """Configuration for FSD50K dataset."""

    # Audio processing parameters (matching UrbanSound8K for consistency)
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    top_db: int = 80
    fixed_length: int = 2048

    # Processing parameters
    batch_size: int = 32
    num_workers: int = 4

    # Default paths
    metadata_dir: Path = Path("data/FSD50K/FSD50K.metadata")
    ground_truth_dir: Path = Path("data/FSD50K/FSD50K.ground_truth")
    _audio_root: Path = Path("data/FSD50K/FSD50K.dev_audio")
    output_dir: Path = Path("data/all_specs")

    # Specific files
    vocabulary_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/vocabulary.csv")
    dev_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/dev.csv")
    _dataset_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/dev.csv")  # Common interface
    eval_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/eval.csv")

    # Cached vocabulary to avoid repeated loading
    _vocabulary: dict[int, str] | None = None
    _name_to_id: dict[str, int] | None = None

    # === Core Dataset Properties ===
    @property
    def audio_root(self) -> Path:
        """Root directory containing audio files."""
        return self._audio_root

    @property
    def dataset_csv(self) -> Path:
        """Path to the main dataset CSV file."""
        return self._dataset_csv

    @property
    def vocabulary(self) -> dict[int, str]:
        """Load and cache FSD50K vocabulary mapping."""
        if self._vocabulary is None:
            self._vocabulary = load_fsd50k_vocabulary(self.vocabulary_csv)
        return self._vocabulary

    @property
    def name_to_id(self) -> dict[str, int]:
        """Load and cache class name to ID mapping."""
        if self._name_to_id is None:
            self._name_to_id = {name: class_id for class_id, name in self.vocabulary.items()}
        return self._name_to_id

    def list_classes(self) -> None:
        """Print all available FSD50K classes."""
        print("FSD50K Classes (200 total):")
        print("=" * 40)
        for class_id in sorted(self.vocabulary.keys()):
            name = self.vocabulary[class_id]
            print(f"{class_id:3d}: {name}")

    # === Metadata and File Management ===

    # === Split Management ===
    def get_available_splits(self) -> list[str]:
        """Return list of valid split names for FSD50K dataset."""
        return ["dev", "eval"]

    def get_default_split(self) -> str:
        """Return the default split name for FSD50K dataset."""
        return "dev"

    def _load_metadata_for_split(self, split: str) -> list[dict]:
        """Load metadata for specified split."""
        csv_path = self.dev_csv if split == "dev" else self.eval_csv

        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        audio_files = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = self.parse_metadata_row(row)
                audio_files.append(parsed)

        return audio_files

    def parse_metadata_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format."""
        labels = row["labels"].split(",")
        mids = row["mids"].split(",")

        return {
            "filename": row["fname"],
            "labels": labels,  # This contains class names, not IDs
            "mids": mids,
            "split": row.get("split", "eval"),  # eval.csv doesn't have split column
            "audio_path": self.get_audio_path(row["fname"]),
        }

    def get_audio_path(self, filename: str, fold: int | None = None) -> Path:
        """Get full path to audio file."""
        if filename.endswith(".wav"):
            return self.audio_root / filename
        return self.audio_root / f"{filename}.wav"

    def get_spectrogram_path(self, filename: str, specs_dir: Path) -> Path:
        """Get path where spectrogram should be stored."""
        spec_filename = filename.replace(".wav", "") + ".pt"
        return specs_dir / spec_filename

    # === Audio Processing ===

    def get_audio_processing_params(self) -> dict[str, Any]:
        """Get audio processing parameters for spectrogram generation."""
        return {
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "top_db": self.top_db,
            "fixed_length": self.fixed_length,
        }

    def create_spectrogram_transform(self):
        """Create PyTorch transform pipeline for generating spectrograms."""
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            ),
            LogNormalize(top_db=self.top_db),
        )

    def get_output_shape(self) -> tuple[int, ...]:
        """Get the shape of tensors produced by this dataset."""
        return (self.n_mels, -1)  # -1 indicates variable time dimension

    def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Fix spectrogram length by cropping outliers but preserving natural variation."""
        current_length = spec.shape[-1]  # Time dimension is last
        max_length = self.fixed_length  # Use as maximum, not target

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

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert stereo to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Create spectrogram
            spec_transform = self.create_spectrogram_transform()
            spec = spec_transform(waveform)

            # Store original length before fixing
            original_length = spec.shape[-1]

            # Fix spectrogram length
            spec = self.fix_spectrogram_length(spec)

            # Save spectrogram
            output_filename = f"{filename}.pt"
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
                positive_class in self.name_to_id
                and self.name_to_id.get(class_name) == positive_class
            )
        return False

    def get_binary_label(
        self, item: dict[str, Any], positive_class_id: int, positive_class_name: str
    ) -> int:
        """Get binary label for an item based on positive class criteria."""
        # item["labels"] already contains class names, not IDs
        return 1 if positive_class_name in item["labels"] else 0
