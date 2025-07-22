"""
Path utilities for AudioLoop.

Simple utility functions to eliminate hardcoded path duplication
and make directory locations configurable via environment variables.
"""

import os
import re
from pathlib import Path


def get_data_root() -> Path:
    """Get the data root directory."""
    return Path(os.getenv("AUDIOLOOP_DATA_ROOT", "data"))


def get_output_root() -> Path:
    """Get the output root directory."""
    return Path(os.getenv("AUDIOLOOP_OUTPUT_ROOT", "."))


def get_specs_dir() -> Path:
    """Get the spectrograms directory."""
    data_root = get_data_root()
    specs_subdir = os.getenv("AUDIOLOOP_SPECS_DIR", "all_specs")
    return data_root / specs_subdir


def get_output_dir(experiment_name: str | None = None) -> Path:
    """Get the outputs directory, with optional experiment suffix."""
    output_root = get_output_root()
    if experiment_name:
        return output_root / f"outputs_{experiment_name}"
    return output_root / "outputs"


def get_training_sets_dir(experiment_name: str | None = None) -> Path:
    """Get the training sets directory, with optional experiment suffix."""
    output_root = get_output_root()
    if experiment_name:
        return output_root / f"training_sets_{experiment_name}"
    return output_root / "training_sets"


def create_output_directories(experiment_name: str | None = None) -> None:
    """Create output and training set directories if they don't exist."""
    get_output_dir(experiment_name).mkdir(parents=True, exist_ok=True)
    get_training_sets_dir(experiment_name).mkdir(parents=True, exist_ok=True)
    get_specs_dir().mkdir(parents=True, exist_ok=True)


def extract_version_from_filename(filepath: Path, file_type: str) -> int | None:
    """Extract version number from versioned filenames."""
    patterns = {
        "model": r"model_v(\d+)\.pt",
        "predictions": r"predictions_v(\d+)\.csv",
        "candidates": r"labeling_candidates_v(\d+)\.csv",
        "training_set": r"training_set_v(\d+)\.csv",
        "binary_labels": r"binary_labels_v(\d+)\.csv",
    }

    if file_type not in patterns:
        raise ValueError(f"Unknown file type: {file_type}. Supported: {list(patterns.keys())}")

    match = re.search(patterns[file_type], str(filepath))
    return int(match.group(1)) if match else None


def audio_to_spec_filename(audio_filename: str) -> str:
    """Convert audio filename to spectrogram filename."""
    return f"{Path(audio_filename).stem}.pt"


def spec_to_audio_filename(spec_filename: str) -> str:
    """Convert spectrogram filename to audio filename (for playback)."""
    return f"{Path(spec_filename).stem}.wav"
