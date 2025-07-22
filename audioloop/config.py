"""
Unified configuration system for AudioLoop.

This module provides centralized configuration management that eliminates
hardcoded paths and coordinates settings across all AudioLoop components.

Key Features:
- Automatic experiment organization with directory suffixes
- Environment-configurable paths (AUDIOLOOP_DATA_ROOT, AUDIOLOOP_OUTPUT_ROOT, etc.)
- Proper configuration precedence: explicit params > env vars > defaults
- Versioned file path generation for models, predictions, training sets
- Dataset-agnostic configuration with extensible registry system
- Path utilities eliminating hardcoded path duplication

Basic Usage:
    from audioloop.config import AudioLoopConfig

    # Create configuration
    config = AudioLoopConfig(experiment_name="test", dataset="urbansound8k")

    # Access organized paths
    config.output_dir          # outputs_test/
    config.training_sets_dir   # training_sets_test/
    config.specs_dir          # data/all_specs/

    # Generate versioned paths
    config.get_model_path(1)        # outputs_test/model_v1.pt
    config.get_predictions_path(1)  # outputs_test/predictions_v1.csv

Configuration Precedence:
    1. Explicit constructor parameters (highest priority)
    2. Environment variables (fallback when no explicit value)
    3. Default values (lowest priority)

Environment Variables:
    AUDIOLOOP_DATASET: Default dataset ('fsd50k' or 'urbansound8k')
    AUDIOLOOP_DATA_ROOT: Root directory for data files (default: 'data')
    AUDIOLOOP_OUTPUT_ROOT: Root directory for outputs (default: '.')
    AUDIOLOOP_SPECS_DIR: Spectrograms subdirectory (default: 'all_specs')
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .datasets.dataset_config import DatasetConfig

from .datasets.dataset_config import DatasetConfig
from .utils.paths import (
    create_output_directories,
    get_output_dir,
    get_specs_dir,
    get_training_sets_dir,
)

# Registry with better typing - avoid runtime import of DatasetConfig
DATASET_CONFIGS: dict[str, type["DatasetConfig"] | None] = {
    "urbansound8k": None,  # Lazy loaded
    "fsd50k": None,        # Lazy loaded
}

DATASET_PROCESSORS: dict[str, type[Any] | None] = {
    "urbansound8k": None,  # Lazy loaded
    "fsd50k": None,        # Lazy loaded
}

# Track loading state to avoid repeated imports
_CLASSES_LOADED = False


def _load_dataset_classes() -> None:
    """Lazy load dataset classes to avoid circular imports."""
    global _CLASSES_LOADED
    if _CLASSES_LOADED:
        return

    from .datasets.fsd50k import FSD50KConfig, FSD50KProcessor
    from .datasets.urbansound8k import UrbanSound8KConfig, UrbanSound8KProcessor

    DATASET_CONFIGS["fsd50k"] = FSD50KConfig
    DATASET_CONFIGS["urbansound8k"] = UrbanSound8KConfig
    DATASET_PROCESSORS["fsd50k"] = FSD50KProcessor
    DATASET_PROCESSORS["urbansound8k"] = UrbanSound8KProcessor

    _CLASSES_LOADED = True


@dataclass
class AudioLoopConfig:
    """Unified configuration for AudioLoop experiments and workflows."""

    # Experiment identification
    experiment_name: str | None = None

    # Dataset configuration (environment variable support)
    dataset: str = field(default_factory=lambda: os.getenv("AUDIOLOOP_DATASET", "fsd50k"))

    # Training parameters (experiment configuration)
    max_epochs: int = 1000
    seed: int = 42
    batch_size: int = 32
    learning_rate: float = 0.001
    model_type: str = "soundcnn"
    use_batchnorm: bool | None = None  # None = auto-detect based on dataset size

    # Stopping criteria configuration
    stopping_criterion_type: str = "plateau"
    patience: int = 20
    min_delta: float = 0.01
    accuracy_floor: float | None = None

    # Active learning parameters (experiment configuration)
    total_candidates: int = 50
    positive_percentage: float = 0.75
    min_confidence: float = 0.8
    selection_mode: str = "confidence"

    # Selection strategy configuration
    basic_transition_f1_threshold: float = 0.2
    basic_transition_confidence_threshold: float = 0.9
    basic_transition_variance_threshold: float = 0.12
    auto_thresholds: bool = False
    estimated_positive_pct: float | None = None

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure dataset classes are loaded
        _load_dataset_classes()

        # Validate dataset is supported
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset}. Supported: {list(DATASET_CONFIGS.keys())}")
        
        # Validate training parameters
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive") 
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.stopping_criterion_type not in ["plateau", "accuracy"]:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_type}")
        if self.stopping_criterion_type == "plateau" and self.patience <= 0:
            raise ValueError("patience must be positive for plateau criterion")
        
        # Validate active learning parameters
        if self.total_candidates <= 0:
            raise ValueError("total_candidates must be positive")
        if not (0.0 <= self.positive_percentage <= 1.0):
            raise ValueError("positive_percentage must be between 0.0 and 1.0")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.selection_mode not in ["confidence", "entropy", "basic_transition"]:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")
        
        # Validate selection strategy parameters
        if not (0.0 <= self.basic_transition_f1_threshold <= 1.0):
            raise ValueError("basic_transition_f1_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.basic_transition_confidence_threshold <= 1.0):
            raise ValueError("basic_transition_confidence_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.basic_transition_variance_threshold <= 1.0):
            raise ValueError("basic_transition_variance_threshold must be between 0.0 and 1.0")
        if self.estimated_positive_pct is not None and not (0.0 <= self.estimated_positive_pct <= 1.0):
            raise ValueError("estimated_positive_pct must be between 0.0 and 1.0")

    @property
    def output_dir(self) -> Path:
        """Get the outputs directory for this configuration."""
        return get_output_dir(self.experiment_name)

    @property
    def training_sets_dir(self) -> Path:
        """Get the training sets directory for this configuration."""
        return get_training_sets_dir(self.experiment_name)

    @property
    def specs_dir(self) -> Path:
        """Get the spectrograms directory."""
        return get_specs_dir()


    def get_model_path(self, version: int) -> Path:
        """Get path for a model file."""
        return self.output_dir / f"model_v{version}.pt"

    def get_predictions_path(self, version: int) -> Path:
        """Get path for a predictions file."""
        return self.output_dir / f"predictions_v{version}.csv"

    def get_candidates_path(self, version: int) -> Path:
        """Get path for a labeling candidates file."""
        return self.output_dir / f"labeling_candidates_v{version}.csv"

    def get_training_set_path(self, version: int) -> Path:
        """Get path for a training set file."""
        return self.training_sets_dir / f"training_set_v{version}.csv"

    def get_binary_labels_path(self, version: int) -> Path:
        """Get path for a binary labels file."""
        return self.output_dir / f"binary_labels_v{version}.csv"

    def get_dataset_config(self) -> DatasetConfig:
        """Get the dataset configuration for the current dataset."""
        _load_dataset_classes()

        config_class = DATASET_CONFIGS.get(self.dataset)
        if config_class is None:
            raise ValueError(
                f"Dataset config not available for '{self.dataset}'. "
                f"Supported datasets: {list(DATASET_CONFIGS.keys())}. "
                f"This may indicate a loading error or unsupported dataset."
            )

        return config_class()

    def get_dataset_processor(self):
        """Get the dataset processor for the current dataset."""
        _load_dataset_classes()
        dataset_config = self.get_dataset_config()

        processor_class = DATASET_PROCESSORS.get(self.dataset)
        if processor_class is None:
            raise ValueError(
                f"Dataset processor not available for '{self.dataset}'. "
                f"Supported datasets: {list(DATASET_PROCESSORS.keys())}. "
                f"This may indicate a loading error or unsupported dataset."
            )

        return processor_class(dataset_config)

    def is_registry_loaded(self) -> bool:
        """Check if dataset registry is properly loaded for debugging."""
        return _CLASSES_LOADED and all(
            DATASET_CONFIGS.get(dataset) is not None
            for dataset in ["fsd50k", "urbansound8k"]
        )

    def create_directories(self) -> None:
        """Create all necessary directories for this configuration."""
        create_output_directories(self.experiment_name)

    def validate_setup(self) -> dict[str, bool]:
        """Validate that the configuration is usable."""
        results = {}

        # Test if we can create output directories
        try:
            self.create_directories()
            results["can_create_outputs"] = True
        except Exception:
            results["can_create_outputs"] = False

        results["specs_dir_exists"] = self.specs_dir.exists()

        return results
