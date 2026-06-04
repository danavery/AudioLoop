import contextlib
import csv
import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class SpectrogramDataset(torch.utils.data.Dataset):
    """
    Unified dataset for loading precomputed spectrograms from CSV files.

    Requires CSV files with headers. Required columns: filename, label
    Optional columns: original_class, audio_path

    The filename column should contain just the audio filename (e.g., "audio_123.flac"),
    not a full path. The actual spec file path is determined by the specs_dir parameter.
    """

    def __init__(self, csv_file, specs_dir="data/specs", dataset_config=None):
        """
        Initialize the dataset.

        Args:
            csv_file: Path to CSV file containing labels with headers
            specs_dir: Directory containing precomputed .pt spectrogram files
            dataset_config: DatasetConfig for spectrogram path resolution and lazy generation.
                           Used to resolve spectrogram paths via get_spectrogram_path() and
                           to generate spectrograms on-the-fly when missing (if audio_path
                           is present in CSV).
        """
        if dataset_config is None:
            raise ValueError(
                "dataset_config is required. SpectrogramDataset uses it for spectrogram "
                "path resolution (get_spectrogram_path) and lazy generation."
            )
        self.specs_dir = specs_dir
        self.dataset_config = dataset_config
        self.samples = []

        # Load from CSV file
        self._load_from_csv(csv_file)

    def _load_from_csv(self, csv_file):
        """Load data from CSV file. Requires headers."""
        with open(csv_file) as f:
            # Validate that file has required headers
            first_line = f.readline().strip()
            f.seek(0)  # Reset to beginning

            first_line_lower = first_line.lower()

            # Must have filename column
            has_filename_column = "filename" in first_line_lower
            # Must have label column
            has_label_column = "label" in first_line_lower

            if not has_filename_column:
                raise ValueError(
                    f"CSV file must have 'filename' column. First line was: {first_line[:100]}"
                )

            if not has_label_column:
                raise ValueError(
                    f"CSV file must have 'label' column. First line was: {first_line[:100]}"
                )

            # Use DictReader for files with headers
            reader = csv.DictReader(f)
            for row in reader:
                sample = self._parse_dict_row(row)
                if sample:
                    self.samples.append(sample)

    def _parse_dict_row(self, row):
        """Parse a row from a CSV with headers."""
        # Get filename (guaranteed to exist due to header validation)
        filename = os.path.basename(row["filename"])

        # Get label (guaranteed to exist due to header validation)
        label = int(row["label"])

        # Optional fields - original_class can be string (class name) or int (class ID)
        original_class = row.get("original_class", None)
        if original_class is not None and original_class != "":
            # Try to parse as int, otherwise keep as string
            with contextlib.suppress(ValueError):
                original_class = int(original_class)
        else:
            original_class = None

        audio_path = row.get("audio_path", None)

        # Build spectrogram path using dataset config (same method used by create_specs)
        spec_filepath = str(self.dataset_config.get_spectrogram_path(filename, Path(self.specs_dir)))

        return {
            "filename": filename,
            "spec_filepath": spec_filepath,
            "label": label,
            "original_class": original_class,
            "audio_path": audio_path,
        }

    def _generate_spec_from_audio(self, audio_path):
        """Generate spectrogram from audio file using dataset config.

        Args:
            audio_path: Path to audio file (str or Path)

        Returns:
            torch.Tensor: Generated spectrogram

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Check minimum file size if configured (catches corrupt files)
        min_size = self.dataset_config.min_audio_file_size
        if min_size is not None:
            file_size = audio_path.stat().st_size
            if file_size < min_size:
                raise RuntimeError(
                    f"Audio file too small ({file_size} bytes, min {min_size}): {audio_path}. "
                    f"Likely corrupt (e.g., FLAC header with no data)."
                )

        # Produce the feature tensor (load -> transform -> fix)
        try:
            return self.dataset_config.extract_one(audio_path)
        except Exception as e:
            # Corrupt or unsupported audio file
            raise RuntimeError(f"Failed to extract features from {audio_path}: {e}") from e

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        spec_filepath = sample["spec_filepath"]

        # Debug mode: log before processing (set AUDIOLOOP_DEBUG_FILES=1 to enable)
        if os.getenv("AUDIOLOOP_DEBUG_FILES"):
            # Use print with flush=True to ensure immediate output (bypasses logging buffering)
            print(f"DEBUG [{idx}] Loading: {sample['filename']}", file=sys.stderr, flush=True)

        try:
            # Try to load cached spectrogram
            if os.path.exists(spec_filepath):
                data = torch.load(spec_filepath)
            else:
                # Try lazy generation if audio_path available
                audio_path = sample.get("audio_path")
                if audio_path:
                    # Debug mode: log audio file being loaded for lazy generation
                    if os.getenv("AUDIOLOOP_DEBUG_FILES"):
                        logger.info(
                            f"[{idx}] Generating from audio: {os.path.basename(audio_path)}"
                        )
                        sys.stderr.flush()  # Flush immediately to see which file crashes
                    # Generate spectrogram from audio
                    data = self._generate_spec_from_audio(audio_path)

                    # Cache to disk for future use
                    os.makedirs(os.path.dirname(spec_filepath), exist_ok=True)
                    torch.save(data, spec_filepath)
                else:
                    # No lazy generation possible - raise error
                    raise FileNotFoundError(
                        f"Spectrogram file not found: {spec_filepath}. "
                        f"Either pre-generate specs with create_specs, or provide "
                        f"audio_path column in CSV for lazy generation."
                    )

            # Build return dictionary with all available fields
            result = {
                "data": data,
                "label": sample["label"],
                "filename": sample["filename"],
                "filepath": spec_filepath,
                "audio_path": sample.get("audio_path"),
            }

            # Only include original_class if it exists
            if sample["original_class"] is not None:
                result["original_class"] = sample["original_class"]

            return result

        except Exception as e:
            # File missing, corrupted, or failed to decode - skip this sample
            # Log at info level (only visible with --verbose)
            import logging

            logging.info(f"Skipping file {sample['filename']}: {e}")
            # Return None to signal skip (collate_fn will filter these out)
            return None
