"""
Path utilities for AudioLoop.

Simple utility functions to eliminate hardcoded path duplication
and make directory locations configurable via environment variables.

Project root detection looks for audioloop.yaml in the current working
directory. This file is created by `python -m audioloop.init_project`.
"""

import os
import re
from pathlib import Path

# Module-level cache to avoid repeated filesystem checks
_project_root_cache: Path | None = None


def clear_project_root_cache() -> None:
    """Clear the cached project root. Useful for testing."""
    global _project_root_cache
    _project_root_cache = None


def get_project_root() -> Path:
    """Get the AudioLoop project root directory.

    Looks for audioloop.yaml in the current working directory.

    Returns:
        Absolute path to the project root directory.

    Raises:
        RuntimeError: If no audioloop.yaml found and no env var set.
    """
    global _project_root_cache

    if _project_root_cache is not None:
        return _project_root_cache

    # Check environment variable override first
    if env_root := os.getenv("AUDIOLOOP_PROJECT_ROOT"):
        _project_root_cache = Path(env_root).resolve()
        return _project_root_cache

    # Check current directory for audioloop.yaml
    cwd = Path.cwd()
    if (cwd / "audioloop.yaml").exists():
        _project_root_cache = cwd.resolve()
        return _project_root_cache

    # No project found
    raise RuntimeError(
        "No AudioLoop project found in current directory.\n"
        "Run 'python -m audioloop.init_project' to create one,\n"
        "or set AUDIOLOOP_PROJECT_ROOT environment variable."
    )


def get_data_root() -> Path:
    """Get the data root directory."""
    if env := os.getenv("AUDIOLOOP_DATA_ROOT"):
        return Path(env)
    return get_project_root() / "data"


def get_output_root() -> Path:
    """Get the output root directory."""
    if env := os.getenv("AUDIOLOOP_OUTPUT_ROOT"):
        return Path(env)
    return get_project_root()


def get_feature_cache_dir() -> Path:
    """Get the spectrograms directory."""
    data_root = get_data_root()
    specs_subdir = os.getenv("AUDIOLOOP_SPECS_DIR", "feature_cache")
    return data_root / specs_subdir


def get_output_dir(experiment_name: str | None = None) -> Path:
    """Get the outputs directory, with optional experiment suffix."""
    output_root = get_output_root()
    if experiment_name:
        return output_root / "outputs" / experiment_name
    return output_root / "outputs"


def get_training_sets_dir(experiment_name: str | None = None) -> Path:
    """Get the training sets directory, with optional experiment suffix."""
    output_root = get_output_root()
    if experiment_name:
        return output_root / "training_sets" / experiment_name
    return output_root / "training_sets"


def create_output_directories(experiment_name: str | None = None) -> None:
    """Create output and training set directories if they don't exist."""
    get_output_dir(experiment_name).mkdir(parents=True, exist_ok=True)
    get_training_sets_dir(experiment_name).mkdir(parents=True, exist_ok=True)
    get_feature_cache_dir().mkdir(parents=True, exist_ok=True)


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
