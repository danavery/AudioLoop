import json
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


def load_audioset_ontology(ontology_path: Path) -> dict[str, str]:
    """Load AudioSet ontology mapping from JSON file.

    Args:
        ontology_path: Path to ontology.json file

    Returns:
        Dictionary mapping MID to display_name
    """
    mid_to_name = {}
    with ontology_path.open("r") as f:
        ontology = json.load(f)
        for entry in ontology:
            mid_to_name[entry["id"]] = entry["name"]
    return mid_to_name


@dataclass
class AudiosetConfig(DatasetConfig):
    """Configuration for AudioSet dataset."""

    # Audio processing parameters (matching other datasets for consistency)
    sample_rate: int = 44100
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    top_db: int = 80
    fixed_length: int = 2048  # AudioSet clips are ~10s, so allow longer spectrograms

    # Processing parameters
    batch_size: int = 32
    num_workers: int = 4

    # Default paths - adjust to your mount point
    metadata_dir: Path = Path("/mnt/audioset/audioset/metadata")
    _audio_root: Path = Path("/mnt/audioset/audioset")
    output_dir: Path = Path("data/all_specs")

    # Specific files
    ontology_json: Path = Path("/mnt/audioset/audioset/metadata/ontology.json")
    # AudioSet has multiple CSV files - we'll use the balanced train set by default
    _dataset_csv: Path = Path("audioset_subset_brass_instrument_100000.csv")
    eval_csv: Path = Path("/mnt/audioset/audioset/metadata/eval_segments.csv")
    unbalanced_csv: Path = Path("/mnt/audioset/audioset/metadata/unbalanced_train_segments.csv")

    # Cached ontology to avoid repeated loading
    _ontology: dict[str, str] | None = None
    _name_to_mid: dict[str, str] | None = None
    _current_split: str = "bal_train"  # Track current split for path construction

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
        """Load and cache AudioSet ontology mapping as integer-indexed dict."""
        if self._ontology is None:
            mid_to_name = load_audioset_ontology(self.ontology_json)
            # Convert to integer indexing for compatibility with existing interface
            self._ontology = dict(enumerate(mid_to_name.values()))
        return self._ontology

    @property
    def name_to_id(self) -> dict[str, int]:
        """Load and cache class name to ID mapping."""
        if self._name_to_mid is None:
            mid_to_name = load_audioset_ontology(self.ontology_json)
            # Create name to integer ID mapping
            name_to_id = {name: i for i, name in enumerate(mid_to_name.values())}
            self._name_to_mid = name_to_id
        return self._name_to_mid

    def list_classes(self) -> None:
        """Print all available AudioSet classes."""
        print("AudioSet Classes (527 total):")
        print("=" * 40)
        for class_id in sorted(self.vocabulary.keys()):
            name = self.vocabulary[class_id]
            print(f"{class_id:3d}: {name}")

    # === Metadata and File Management ===

    def load_metadata(self, split: str = "dev") -> list[dict]:
        """Load metadata for specified split."""
        # Map standard split names to AudioSet split names
        if split == "dev":
            split = "balanced_train"
        if split == "balanced_train":
            csv_path = self._dataset_csv
            self._current_split = "unbal_train"  # Subset files are from unbalanced dataset
        elif split == "eval":
            csv_path = self.eval_csv
            self._current_split = "eval"
        elif split == "unbalanced_train":
            csv_path = self.unbalanced_csv
            self._current_split = "unbal_train"
        else:
            raise ValueError(
                f"Unknown split: {split}. Use 'balanced_train', 'eval', or 'unbalanced_train'"
            )

        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        audio_files = []
        with csv_path.open("r") as f:
            # Skip header comments (lines starting with #)
            for line in f:
                if not line.startswith("#"):
                    # Parse CSV line manually since it uses custom format
                    parts = [p.strip() for p in line.strip().split(",")]
                    if len(parts) >= 4:
                        parsed = self.parse_metadata_row(
                            {
                                "YTID": parts[0],
                                "start_seconds": parts[1],
                                "end_seconds": parts[2],
                                "positive_labels": parts[3],
                            }
                        )
                        audio_files.append(parsed)

        return audio_files

    def parse_metadata_row(self, row: dict[str, str]) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format."""
        ytid = row["YTID"]
        start_seconds = float(row["start_seconds"])
        end_seconds = float(row["end_seconds"])

        # Parse positive labels - they're quoted strings with MIDs separated by commas
        positive_labels_str = row["positive_labels"].strip('"')
        label_mids = [mid.strip() for mid in positive_labels_str.split(",") if mid.strip()]

        # Convert MIDs to display names using cached ontology
        if not hasattr(self, "_mid_to_name"):
            self._mid_to_name = load_audioset_ontology(self.ontology_json)

        label_names = []
        for mid in label_mids:
            if mid in self._mid_to_name:
                label_names.append(self._mid_to_name[mid])

        filename = f"{ytid}.flac"

        return {
            "filename": filename,
            "ytid": ytid,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "labels": label_names,  # Display names for consistency with other datasets
            "mids": label_mids,  # Keep original MIDs for reference
            "audio_path": self.get_audio_path(filename),
        }

    def get_audio_path(self, filename: str, fold: int | None = None) -> Path:
        """Get full path to audio file.

        AudioSet organizes files in subdirectories by first 2 characters of YTID.
        """
        ytid = filename.split("_")[0]  # Extract YTID from filename

        # Determine which subdirectory based on first 2 characters
        first_two = ytid[:2]

        # Use the current split (set during load_metadata) to construct path
        # File existence is checked later in process_single_file
        return self.audio_root / self._current_split / first_two / filename

    def get_spectrogram_path(self, filename: str, specs_dir: Path) -> Path:
        """Get path where spectrogram should be stored."""
        spec_filename = filename.replace(".flac", "") + ".pt"
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
                # Don't log individual missing files to avoid disrupting tqdm progress bar
                # Failure count will be shown in progress bar postfix and final summary
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
            output_filename = filename.replace(".flac", "") + ".pt"
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
        """Get binary label for an item based on positive class criteria.

        For AudioSet's multilabel format, returns 1 if positive_class_name
        appears in the item's labels list, 0 otherwise.
        """
        # item["labels"] contains display names for this sample
        return 1 if positive_class_name in item["labels"] else 0
