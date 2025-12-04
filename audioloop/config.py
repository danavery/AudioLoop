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
    config.output_dir          # outputs/test/
    config.training_sets_dir   # training_sets/test/
    config.specs_dir          # data/all_specs/

    # Generate versioned paths
    config.get_model_path(1)        # outputs/test/model_v1.pt
    config.get_predictions_path(1)  # outputs/test/predictions_v1.csv

Class Weighting Modes:
    # No weighting (default) - treats all classes equally
    config = AudioLoopConfig(class_weighting=None)

    # Adaptive weighting - calculates inverse frequency from training set each cycle
    config = AudioLoopConfig(class_weighting="adaptive")

    # Fixed weighting - maintains consistent target positive ratio across cycles
    config = AudioLoopConfig(class_weighting=0.25)  # Target 25% positive samples

Configuration Precedence:
    1. Explicit constructor parameters (highest priority)
    2. Environment variables (fallback when no explicit value)
    3. Default values (lowest priority)

Environment Variables:
    AUDIOLOOP_EXPERIMENT: Default experiment name (default: None, uses base directories)
    AUDIOLOOP_DATASET: Default dataset ('fsd50k' or 'urbansound8k')
    AUDIOLOOP_DATA_ROOT: Root directory for data files (default: 'data')
    AUDIOLOOP_OUTPUT_ROOT: Root directory for outputs (default: '.')
    AUDIOLOOP_SPECS_DIR: Spectrograms subdirectory (default: 'all_specs')
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datasets.dataset_config import DatasetConfig
from .utils.paths import (
    create_output_directories,
    get_output_dir,
    get_specs_dir,
    get_training_sets_dir,
)

# Dynamic dataset discovery - no hardcoded registry needed


@dataclass
class AudioLoopConfig:
    """Unified configuration for AudioLoop experiments and workflows."""

    # Experiment identification (environment variable support)
    experiment_name: str | None = field(default_factory=lambda: os.getenv("AUDIOLOOP_EXPERIMENT"))

    # Dataset configuration (environment variable support)
    dataset: str = field(default_factory=lambda: os.getenv("AUDIOLOOP_DATASET", "fsd50k"))
    subset_csv: Path | None = None  # Optional subset CSV to restrict dataset files

    # Training parameters (experiment configuration)
    max_epochs: int = 1000
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2  # Number of parallel data loading workers
    learning_rate: float = 0.001
    weight_decay: float = 1e-5  # L2 regularization for optimizer

    # Learning rate scheduler parameters
    use_lr_scheduler: bool = True  # Enable ReduceLROnPlateau scheduler
    lr_scheduler_factor: float = 0.5  # Multiply LR by this on plateau
    lr_scheduler_patience: int = 5  # Epochs to wait before reducing LR
    lr_scheduler_min_lr: float = 1e-6  # Minimum learning rate

    model_type: str = "cnn5layer"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    class_weighting: str | float | None = (
        None  # "adaptive", float (target positive ratio 0.0-1.0), or None
    )

    # Stopping criteria configuration (within-epoch)
    stopping_criterion_type: str = "plateau"
    patience: int = 20
    min_delta: float = 0.01
    accuracy_floor: float | None = None

    # Cross-cycle stopping criteria (for active learning loops)
    cycle_stopping_strategy: str = "none"  # 'label', 'search', or 'none'
    cycle_patience: int = 5
    cycle_min_delta: float = 0.02
    cycle_min_cycles: int = 10
    cycle_window: int = 3
    cycle_std_threshold: float = 0.08
    precision_floor: float | str = "auto"  # Only used for 'search' mode

    # Active learning parameters (experiment configuration)
    total_candidates: int = 50
    positive_percentage: float | None = None  # None = no stratification (pure entropy)
    min_confidence: float = 0.8
    selection_mode: str = "entropy"

    # Selection strategy configuration
    basic_transition_f1_threshold: float = 0.2
    basic_transition_confidence_threshold: float = 0.9
    basic_transition_variance_threshold: float = 0.12
    auto_thresholds: bool = False
    estimated_positive_pct: float | None = None

    # Ground truth evaluation configuration
    with_ground_truth: bool = False  # Include ground truth columns in predictions CSV

    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_dataset()
        self._validate_training_params()
        self._validate_active_learning_params()
        self._validate_selection_strategy_params()

    def _validate_dataset(self):
        """Validate that the dataset is supported."""
        from .datasets.dataset_registry import list_available_datasets

        available = list_available_datasets()
        if self.dataset not in available:
            raise ValueError(
                f"Unknown dataset: {self.dataset}. Supported: {', '.join(sorted(available))}"
            )

    def _validate_training_params(self):
        """Validate core training parameters."""
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.lr_scheduler_factor <= 0 or self.lr_scheduler_factor >= 1:
            raise ValueError("lr_scheduler_factor must be between 0 and 1 (exclusive)")
        if self.lr_scheduler_patience <= 0:
            raise ValueError("lr_scheduler_patience must be positive")
        if self.lr_scheduler_min_lr <= 0 or self.lr_scheduler_min_lr >= self.learning_rate:
            raise ValueError("lr_scheduler_min_lr must be positive and less than learning_rate")
        if self.stopping_criterion_type not in ["plateau", "accuracy"]:
            raise ValueError(f"Unknown stopping criterion: {self.stopping_criterion_type}")
        if self.stopping_criterion_type == "plateau" and self.patience <= 0:
            raise ValueError("patience must be positive for plateau criterion")
        if isinstance(self.class_weighting, float) and not (0.0 < self.class_weighting < 1.0):
            raise ValueError("class_weighting must be between 0.0 and 1.0 (exclusive)")
        if isinstance(self.class_weighting, str) and self.class_weighting != "adaptive":
            raise ValueError("class_weighting must be 'adaptive', a float, or None")

    def _validate_active_learning_params(self):
        """Validate active learning cycle parameters."""
        if self.total_candidates <= 0:
            raise ValueError("total_candidates must be positive")
        if self.positive_percentage is not None and not (0.0 <= self.positive_percentage <= 1.0):
            raise ValueError("positive_percentage must be None or between 0.0 and 1.0")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if self.selection_mode not in [
            "confidence",
            "entropy",
            "mixed_entropy",
            "basic_transition",
            "stratified_uncertainty",
            "random",
        ]:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

    def _validate_selection_strategy_params(self):
        """Validate parameters for specific selection strategies."""
        # Validate mutual exclusivity: mixed_entropy and positive_percentage
        if self.selection_mode == "mixed_entropy" and self.positive_percentage is not None:
            raise ValueError(
                "selection_mode='mixed_entropy' is incompatible with positive_percentage stratification. "
                "Set positive_percentage=None to use mixed entropy sampling."
            )

        if not (0.0 <= self.basic_transition_f1_threshold <= 1.0):
            raise ValueError("basic_transition_f1_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.basic_transition_confidence_threshold <= 1.0):
            raise ValueError("basic_transition_confidence_threshold must be between 0.0 and 1.0")
        if not (0.0 <= self.basic_transition_variance_threshold <= 1.0):
            raise ValueError("basic_transition_variance_threshold must be between 0.0 and 1.0")
        if self.estimated_positive_pct is not None and not (
            0.0 <= self.estimated_positive_pct <= 1.0
        ):
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

    def get_inference_csv_path(self, dataset_name: str) -> Path:
        """Get path for the inference CSV file."""
        return self.output_dir / f"{dataset_name}_files.csv"

    def get_dataset_config(self) -> DatasetConfig:
        """Get the dataset configuration for the current dataset.

        If subset_csv is specified, automatically applies it to restrict the dataset files.
        """
        from .datasets.dataset_registry import get_dataset_config_class

        config_class = get_dataset_config_class(self.dataset)
        dataset_config = config_class()

        # Apply subset restriction if specified
        if self.subset_csv:
            if not self.subset_csv.exists():
                raise FileNotFoundError(f"Subset CSV not found: {self.subset_csv}")
            dataset_config.set_custom_csv(self.subset_csv)

        return dataset_config

    def create_directories(self) -> None:
        """Create all necessary directories for this configuration."""
        create_output_directories(self.experiment_name)
