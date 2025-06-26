import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch import nn

from ..utils.log_normalize import LogNormalize

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
    9: "street_music"
}

# Reverse mapping for convenience
CLASS_NAME_TO_ID = {name: class_id for class_id, name in URBANSOUND8K_CLASSES.items()}


def get_class_name(class_id: int) -> str:
    """Get human-readable class name from UrbanSound8K class ID.

    Args:
        class_id: UrbanSound8K class ID (0-9)

    Returns:
        Human-readable class name

    Raises:
        ValueError: If class_id is not valid
    """
    if class_id not in URBANSOUND8K_CLASSES:
        raise ValueError(f"Invalid class ID: {class_id}. Must be 0-9.")
    return URBANSOUND8K_CLASSES[class_id]


def get_class_id(class_name: str) -> int:
    """Get UrbanSound8K class ID from human-readable class name.

    Args:
        class_name: Human-readable class name

    Returns:
        UrbanSound8K class ID (0-9)

    Raises:
        ValueError: If class_name is not valid
    """
    if class_name not in CLASS_NAME_TO_ID:
        valid_names = list(CLASS_NAME_TO_ID.keys())
        raise ValueError(f"Invalid class name: '{class_name}'. Valid names: {valid_names}")
    return CLASS_NAME_TO_ID[class_name]


def list_classes() -> None:
    """Print all available UrbanSound8K classes."""
    print("UrbanSound8K Classes:")
    print("=" * 30)
    for class_id, name in URBANSOUND8K_CLASSES.items():
        print(f"{class_id}: {name}")


@dataclass
class UrbanSound8KConfig:
    """Configuration for UrbanSound8K dataset."""

    # Audio processing parameters
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    top_db: int = 80
    fixed_length: int = 993  # Max length from training data analysis

    # Processing parameters
    batch_size: int = 32
    num_workers: int = 4

    # Default paths
    metadata_csv: Path = Path("data/urbansound8k/UrbanSound8K.csv")
    audio_root: Path = Path("data/urbansound8k")
    output_dir: Path = Path("data/all_specs")
    inference_csv: Path = Path("outputs/urbansound8k_files.csv")


class UrbanSound8KProcessor:
    """Handles UrbanSound8K-specific dataset operations."""

    def __init__(self, config: UrbanSound8KConfig):
        self.config = config
        self.spec_transform = self._create_transform()

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

    def get_audio_path(self, filename: str, fold: int) -> Path:
        """Get full path to audio file given filename and fold."""
        return self.config.audio_root / f"fold{fold}" / filename

    def parse_metadata_row(self, row: dict[str, str]) -> dict:
        """Parse a metadata CSV row into standardized format."""
        return {
            "filename": row["slice_file_name"],
            "class_id": int(row["classID"]),
            "class_name": row["class"],
            "fold": int(row["fold"]),
        }

    def load_metadata(self) -> list[dict]:
        """Load all metadata for the UrbanSound8K dataset."""
        if not self.config.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.config.metadata_csv}")

        audio_files = []
        with self.config.metadata_csv.open("r") as f:
            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                parsed = self.parse_metadata_row(row)
                parsed["audio_path"] = self.get_audio_path(parsed["filename"], parsed["fold"])
                audio_files.append(parsed)

        return audio_files

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
            output_filename = filename.replace(".wav", ".pt")
            output_path = output_dir / output_filename
            torch.save(spec, output_path)

            return True

        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            return False
