import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch import nn

from audioloop.utils.log_normalize import LogNormalize

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
class FSD50KConfig:
    """Configuration for FSD50K dataset."""

    # Audio processing parameters (matching UrbanSound8K for consistency)
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    top_db: int = 80
    fixed_length: int = 993  # Using same as UrbanSound8K for consistency

    # Processing parameters
    batch_size: int = 32
    num_workers: int = 4

    # Default paths
    metadata_dir: Path = Path("data/FSD50K/FSD50K.metadata")
    ground_truth_dir: Path = Path("data/FSD50K/FSD50K.ground_truth")
    audio_root: Path = Path("data/FSD50K/FSD50K.dev_audio")
    output_dir: Path = Path("data/all_specs")

    # Specific files
    vocabulary_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/vocabulary.csv")
    dev_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/dev.csv")
    eval_csv: Path = Path("data/FSD50K/FSD50K.ground_truth/eval.csv")


class FSD50KProcessor:
    """Handles FSD50K-specific dataset operations."""

    def __init__(self, config: FSD50KConfig):
        self.config = config
        self.spec_transform = self._create_transform()

        # Load vocabulary for class name/ID conversion
        self.vocabulary = load_fsd50k_vocabulary(self.config.vocabulary_csv)
        self.name_to_id = {name: class_id for class_id, name in self.vocabulary.items()}

    def _create_transform(self) -> nn.Sequential:
        """Create the spectrogram transform pipeline."""
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
            ),
            LogNormalize(top_db=self.config.top_db),
        )

    def get_audio_path(self, filename: str) -> Path:
        """Get full path to audio file given filename."""
        return self.config.audio_root / f"{filename}.wav"

    def parse_metadata_row(self, row: dict[str, str]) -> dict:
        """Parse a metadata CSV row into standardized format."""
        labels = row["labels"].split(",")
        mids = row["mids"].split(",")

        return {
            "filename": row["fname"],
            "labels": labels,
            "mids": mids,
            "split": row.get("split", "eval"),  # eval.csv doesn't have split column
            "audio_path": self.get_audio_path(row["fname"]),
        }

    def load_metadata(self, split: str = "dev") -> list[dict]:
        """Load metadata for specified split.

        Args:
            split: 'dev' for development set or 'eval' for evaluation set

        Returns:
            List of parsed metadata dictionaries
        """
        csv_path = self.config.dev_csv if split == "dev" else self.config.eval_csv

        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        audio_files = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = self.parse_metadata_row(row)
                audio_files.append(parsed)

        return audio_files

    def create_binary_labels_one_vs_all(
        self,
        positive_class: str,
        split: str = "dev",
        output_csv: str = "outputs/fsd50k_binary_labels.csv",
    ) -> str:
        """Create binary labels using one-vs-all strategy.

        Args:
            positive_class: Class name to treat as positive
            split: Dataset split to use ('dev' or 'eval')
            output_csv: Path for output binary labels CSV

        Returns:
            Path to created binary labels CSV
        """
        # Validate class name
        if positive_class not in self.name_to_id:
            raise ValueError(
                f"Invalid class name: '{positive_class}'. Use list_classes() to see valid names."
            )

        # Load metadata
        metadata = self.load_metadata(split)

        binary_data = []
        positive_count = 0
        negative_count = 0

        for item in metadata:
            filename = item["filename"]
            labels = item["labels"]

            # Binary classification: positive_class vs everything else
            is_positive = 1 if positive_class in labels else 0

            if is_positive:
                positive_count += 1
            else:
                negative_count += 1

            binary_data.append(
                {
                    "filename": filename,
                    "label": is_positive,
                    "original_labels": ",".join(labels),
                    "strategy": f"one_vs_all_{positive_class}",
                    "split": item["split"],
                }
            )

        # Write binary labels CSV
        import os

        os.makedirs("outputs", exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            fieldnames = ["filename", "label", "original_labels", "strategy", "split"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(binary_data)

        negative_class_name = f"not_{positive_class}"
        print(f"Created FSD50K binary labels: {output_csv}")
        print(f"{positive_class} samples: {positive_count}")
        print(f"{negative_class_name} samples: {negative_count}")
        print(f"Total samples: {len(binary_data)}")
        print(f"Split: {split}")

        return output_csv

    def create_binary_labels_semantic_group(
        self,
        group_name: str,
        positive_classes: set[str] | None = None,
        split: str = "dev",
        output_csv: str = "outputs/fsd50k_binary_labels.csv",
    ) -> str:
        """Create binary labels using semantic grouping strategy.

        Args:
            group_name: Name of predefined group or custom name
            positive_classes: Set of class names to treat as positive (if None, uses predefined group)
            split: Dataset split to use ('dev' or 'eval')
            output_csv: Path for output binary labels CSV

        Returns:
            Path to created binary labels CSV
        """
        if positive_classes is None:
            if group_name not in SEMANTIC_GROUPS:
                available_groups = list(SEMANTIC_GROUPS.keys())
                raise ValueError(
                    f"Unknown semantic group: '{group_name}'. Available: {available_groups}"
                )
            positive_classes = SEMANTIC_GROUPS[group_name]

        # Validate class names
        for class_name in positive_classes:
            if class_name not in self.name_to_id:
                raise ValueError(f"Invalid class name in group: '{class_name}'")

        # Load metadata
        metadata = self.load_metadata(split)

        binary_data = []
        positive_count = 0
        negative_count = 0

        for item in metadata:
            filename = item["filename"]
            labels = item["labels"]

            # Binary classification: any positive class vs none
            is_positive = 1 if any(label in positive_classes for label in labels) else 0

            if is_positive:
                positive_count += 1
            else:
                negative_count += 1

            binary_data.append(
                {
                    "filename": filename,
                    "label": is_positive,
                    "original_labels": ",".join(labels),
                    "strategy": f"semantic_group_{group_name}",
                    "split": item["split"],
                }
            )

        # Write binary labels CSV
        import os

        os.makedirs("outputs", exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            fieldnames = ["filename", "label", "original_labels", "strategy", "split"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(binary_data)

        negative_class_name = f"not_{group_name}"
        print(f"Created FSD50K binary labels: {output_csv}")
        print(f"{group_name} samples: {positive_count}")
        print(f"{negative_class_name} samples: {negative_count}")
        print(f"Total samples: {len(binary_data)}")
        print(f"Split: {split}")

        return output_csv

    def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
        """Fix spectrogram to target length by padding or cropping."""
        current_length = spec.shape[-1]  # Time dimension is last
        target_length = self.config.fixed_length

        if current_length < target_length:
            # Pad with zeros on the right
            pad_size = target_length - current_length
            spec = torch.nn.functional.pad(spec, (0, pad_size), mode="constant", value=0)
        elif current_length > target_length:
            # Crop from the center
            start_idx = (current_length - target_length) // 2
            spec = spec[..., start_idx : start_idx + target_length]

        return spec

    def process_single_file(self, file_info: dict, output_dir: Path) -> bool:
        """Process a single audio file and save its spectrogram."""
        try:
            audio_path = file_info["audio_path"]
            filename = file_info["filename"]

            # Check if audio file exists
            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                return False

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Create spectrogram
            spec = self.spec_transform(waveform)

            # Fix spectrogram length
            spec = self.fix_spectrogram_length(spec)

            # Save spectrogram
            output_filename = f"{filename}.pt"
            output_path = output_dir / output_filename
            torch.save(spec, output_path)

            return True

        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            return False
