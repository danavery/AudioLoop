import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audioloop.utils.paths import get_project_root

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

    description = "AudioSet: 527 classes, 10s YouTube clips (multi-label)"

    # Default paths - adjust to your mount point
    _audio_root: Path = Path("/mnt/audioset/audioset")

    # Specific files
    ontology_json: Path = Path("/mnt/audioset/audioset/metadata/ontology.json")
    balanced_csv: Path = Path("/mnt/audioset/audioset/metadata/balanced_train_segments.csv")
    unbalanced_csv: Path = Path("/mnt/audioset/audioset/metadata/unbalanced_train_segments.csv")
    eval_csv: Path = Path("/mnt/audioset/audioset/metadata/eval_segments.csv")
    # Default to balanced train (smallest dataset)
    _dataset_csv: Path = Path("/mnt/audioset/audioset/metadata/balanced_train_segments.csv")

    # Cached ontology to avoid repeated loading
    _ontology: dict[int, str] | None = None
    _name_to_mid: dict[str, int] | None = None

    # Custom CSV support
    _custom_csv_path: Path | None = None

    # === Bad Files Exclusion ===
    def get_bad_files(self) -> set[str]:
        """Return set of filenames known to crash during processing.

        These files have corrupt audio that causes torchaudio to segfault.
        Add new problematic files here as they're discovered.
        This is in addition to (and in this case, somewhat redundant to) the min_audio_file_size check
        """
        return {
            "Z3YaJ9Vi4lY.flac",
            "lld156ULI5U.flac",
            "PXnzhGJctVA.flac",
            "k-g2CuGv12I.flac",  # Corrupt audio causing segfault in brass subset
        }

    @property
    def min_audio_file_size(self) -> int | None:
        """Minimum audio file size - files under 9KB are corrupt FLAC headers."""
        return 9000  # AudioSet clips are ~10s, corrupt files are 8288 bytes

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
        """Audio file extension for AudioSet (.flac)."""
        return ".flac"

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

    # === Split Management ===
    def get_available_splits(self) -> list[str]:
        """Return list of valid split names for AudioSet dataset."""
        return ["bal_train", "unbal_train", "eval"]

    def get_default_split(self) -> str:
        """Return the default split name for AudioSet dataset."""
        return "unbal_train"

    def supports_custom_csv(self) -> bool:
        """AudioSet supports custom CSV files for subsets."""
        return True

    def set_custom_csv(self, custom_csv_path: Path) -> None:
        """Set a custom CSV file to use instead of the default dataset CSV.

        Args:
            custom_csv_path: Path to the custom CSV file

        Raises:
            FileNotFoundError: If the custom CSV file doesn't exist
        """
        if not custom_csv_path.exists():
            raise FileNotFoundError(f"Custom CSV file not found: {custom_csv_path}")

        self._custom_csv_path = custom_csv_path

    def _load_metadata_for_split(self, split: str) -> list[dict]:
        """Load metadata for specified split."""

        # If custom CSV is set, use it regardless of split parameter
        if hasattr(self, "_custom_csv_path") and self._custom_csv_path is not None:
            return self._load_subset_csv(self._custom_csv_path)

        # Map split name to CSV path (split name already matches filesystem directory)
        if split == "bal_train":
            csv_path = self.balanced_csv
        elif split == "unbal_train":
            csv_path = self.unbalanced_csv
        elif split == "eval":
            csv_path = self.eval_csv
        else:
            # This shouldn't happen due to validation in base class, but keep for safety
            raise ValueError(f"Unknown split: {split}. Use 'bal_train', 'unbal_train', or 'eval'")

        dir_split = split  # Split name matches filesystem directory

        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        audio_files = []
        import csv as csv_module

        with csv_path.open("r") as f:
            reader = csv_module.reader(f, skipinitialspace=True)
            for row in reader:
                # Skip header comments (lines starting with #)
                if row and not row[0].startswith("#") and len(row) >= 4:
                    parsed = self.parse_metadata_row(
                        {
                            "YTID": row[0],
                            "start_seconds": row[1],
                            "end_seconds": row[2],
                            "positive_labels": row[3],
                        },
                        split=dir_split,
                    )
                    audio_files.append(parsed)

        return audio_files

    def _load_subset_csv(self, csv_path: Path) -> list[dict]:
        """Load metadata from a subset CSV created by create_subset().

        Subset CSV format: filename,label,original_class,split,audio_path
        """
        import csv

        bad_files = self.get_bad_files()
        audio_files = []
        skipped_bad = 0

        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract filename without extension to get YTID
                filename = row["filename"]

                # Skip known bad files that cause segfaults (fast set lookup)
                if filename in bad_files:
                    skipped_bad += 1
                    continue

                audio_path = Path(row["audio_path"])
                ytid = filename.replace(".flac", "").split("_")[0]

                # Reconstruct metadata format expected by the rest of the system
                audio_files.append(
                    {
                        "filename": filename,
                        "ytid": ytid,
                        "start_seconds": 0.0,  # Not available in subset CSV
                        "end_seconds": 10.0,  # AudioSet default clip length
                        "labels": [row["original_class"]],  # Use original_class as the label
                        "mids": [],  # MIDs not available in subset format
                        "split": row["split"],
                        "audio_path": audio_path,
                    }
                )

        if skipped_bad > 0:
            logger.info(f"Skipped {skipped_bad} known bad files that cause segfaults")

        return audio_files

    def parse_metadata_row(self, row: dict[str, str], split: str | None = None) -> dict[str, Any]:
        """Parse a single CSV row into standardized metadata format.

        Args:
            row: Raw CSV row data
            split: Filesystem directory split (bal_train, unbal_train, eval).
                   If None, defaults to unbal_train.
        """
        if split is None:
            split = "unbal_train"
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
            "split": split,  # Include split in metadata
            "audio_path": self.get_audio_path(filename, split=split),
        }

    def get_audio_path(
        self, filename: str, split: str | None = None, fold: int | None = None
    ) -> Path:
        """Get full path to audio file.

        AudioSet organizes files in subdirectories by first 2 characters of YTID.

        Args:
            filename: Audio filename (e.g., "Y123.flac")
            split: Filesystem directory (bal_train, unbal_train, eval)
            fold: Ignored for AudioSet (no fold structure)

        Returns:
            Full path to audio file
        """
        if split is None:
            # Use default split from configuration
            split = self.get_default_split()

        # Extract YTID from filename (remove .flac extension first)
        ytid = filename.replace(".flac", "")
        first_two = ytid[:2]  # Determine subdirectory based on first 2 characters

        return self.audio_root / split / first_two / filename

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

        Args:
            output_path: Where to write the subset CSV
            class_name: Class name to use as positive class
            max_samples: Maximum total samples (positive + negative)
            positive_ratio: Target ratio of positive samples (0.0-1.0)
            split: Dataset split to sample from (default: unbal_train)
            seed: Random seed for reproducibility

        Returns:
            Path to created CSV file
        """
        import csv
        import random

        random.seed(seed)
        split = split or "unbal_train"

        # Validate class exists
        if class_name not in self.name_to_id:
            raise ValueError(
                f"Class '{class_name}' not found. Use list_classes() to see valid names."
            )

        # Load and filter
        all_metadata = self.load_metadata(split=split)
        positives = [m for m in all_metadata if class_name in m.get("labels", [])]
        negatives = [m for m in all_metadata if class_name not in m.get("labels", [])]

        # Sample requested amounts (missing files handled by lazy loading during training)
        target_pos = min(int(max_samples * positive_ratio), len(positives))
        target_neg = min(max_samples - target_pos, len(negatives))

        sampled_positives = random.sample(positives, target_pos)
        sampled_negatives = random.sample(negatives, target_neg)

        # Combine samples (no file existence check - lazy loading handles missing files)
        samples = [(m, 1, class_name) for m in sampled_positives]
        samples += [(m, 0, m["labels"][0] if m["labels"] else "unknown") for m in sampled_negatives]

        # Calculate actual achieved ratio
        actual_pos_count = len(sampled_positives)
        actual_total = len(samples)
        actual_ratio = actual_pos_count / actual_total if actual_total > 0 else 0

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "original_class", "split", "audio_path"])
            for item, label, orig_class in samples:
                writer.writerow(
                    [
                        item["filename"],
                        label,
                        orig_class,
                        item["split"],
                        str(item["audio_path"]),
                    ]
                )

        # Log summary
        print(f"\nSubset created: {output_path}")
        print(f"  Total samples: {actual_total:,}")
        print(f"  Positive ({class_name}): {actual_pos_count:,} ({actual_ratio:.2%})")
        print(f"  Negative: {len(sampled_negatives):,} ({(1 - actual_ratio):.2%})")
        if abs(actual_ratio - positive_ratio) > 0.01:
            print(
                f"  ⚠ Note: Requested {positive_ratio:.1%} positive, but only {len(positives):,} available in dataset"
            )

        return output_path

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
