import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from audioloop.utils.paths import get_project_root

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
            class_id, class_name, _mid = row
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

    description = "FSD50K: 200 Freesound classes, variable-length clips (multi-label)"

    # Audio processing parameters (matching UrbanSound8K for consistency)
    _sample_rate: int = 44100
    _n_fft: int = 1024
    _hop_length: int = 256
    _n_mels: int = 128
    _top_db: int = 80
    _max_spectrogram_length: int = 2048

    # Default paths
    _audio_root: Path = Path("data/FSD50K")  # Base directory, split determines subdirectory

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
        if self._audio_root.is_absolute():
            return self._audio_root
        # Resolve relative paths against project root
        return get_project_root() / self._audio_root

    @property
    def audio_extension(self) -> str:
        """Audio file extension for FSD50K (.wav)."""
        return ".wav"

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

    def parse_metadata_row(self, row: dict[str, str], split: str | None = None) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format."""
        labels = row["labels"].split(",")
        mids = row["mids"].split(",")

        # Use split from CSV if available, otherwise use provided split parameter
        metadata_split = row.get("split", split or "eval")

        return {
            "filename": row["fname"],
            "labels": labels,  # This contains class names, not IDs
            "mids": mids,
            "split": metadata_split,
            "audio_path": self.get_audio_path(row["fname"], split=metadata_split),
        }

    def get_audio_path(
        self, filename: str, split: str | None = None, fold: int | None = None
    ) -> Path:
        """Get full path to audio file.

        Args:
            filename: Audio filename from metadata
            split: Dataset split (dev or eval) - determines subdirectory
            fold: Ignored for FSD50K (no fold structure)

        Returns:
            Full path to audio file in FSD50K.{split}_audio/ subdirectory
        """
        # Default to dev if no split provided
        if split is None:
            split = self.get_default_split()

        # Construct path: data/FSD50K/FSD50K.dev_audio/filename.wav
        audio_dir = self.audio_root / f"FSD50K.{split}_audio"

        if filename.endswith(".wav"):
            return audio_dir / filename
        return audio_dir / f"{filename}.wav"

    def get_spectrogram_path(self, filename: str, specs_dir: Path) -> Path:
        """Get path where spectrogram should be stored."""
        spec_filename = filename.replace(".wav", "") + ".pt"
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

    def process_single_file(self, file_info: dict, output_dir: Path) -> tuple[bool, int | None]:
        """Process a single audio file and save its spectrogram."""
        try:
            audio_path = file_info["audio_path"]
            filename = file_info["filename"]

            # Check if audio file exists
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return False, None

            # Produce the feature tensor (load -> transform -> fix)
            spec = self.feature_extractor.extract_one(audio_path)
            spec_length = spec.shape[-1]

            # Save spectrogram
            output_filename = f"{filename}.pt"
            output_path = output_dir / output_filename
            torch.save(spec, output_path)

            return True, spec_length

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
