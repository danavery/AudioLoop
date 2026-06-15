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
    # Fixed weighting (default) - prevents model collapse with imbalanced training data
    config = AudioLoopConfig(class_weighting=0.70)  # Target 70% positive weight

    # No weighting - treats all classes equally (can collapse to all-positive)
    config = AudioLoopConfig(class_weighting=None)

    # Adaptive weighting - calculates inverse frequency from training set each cycle
    config = AudioLoopConfig(class_weighting="adaptive")

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
    get_project_root,
    get_specs_dir,
    get_training_sets_dir,
)


@dataclass
class AudioLoopConfig:
    """Unified configuration for AudioLoop experiments and workflows."""

    # Experiment identification (environment variable support)
    experiment_name: str | None = field(default_factory=lambda: os.getenv("AUDIOLOOP_EXPERIMENT"))
    batch_prefix: str | None = None  # Optional prefix for batch runs (only affects output_dir)

    # Dataset configuration (environment variable support)
    dataset: str = field(default_factory=lambda: os.getenv("AUDIOLOOP_DATASET", "fsd50k"))
    subset_csv: Path | None = None  # Optional subset CSV to restrict dataset files
    specs_dir_path: str | None = None  # Spectrogram dir (relative to project root, or absolute)

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

    # Feature extractor selection + parameters (audio->tensor production). Mirrors
    # model_type/model_kwargs: feature_extractor_type picks the extractor class (see
    # get_feature_extractor), feature_extractor_kwargs are its constructor params and are
    # therefore type-dependent (e.g. sample_rate/n_fft/n_mels for "spectrogram"). Empty
    # kwargs = that extractor's defaults (spectrogram = 44.1kHz log-mel).
    feature_extractor_type: str = "spectrogram"
    feature_extractor_kwargs: dict[str, Any] = field(default_factory=dict)
    class_weighting: str | float | None = (
        0.70  # "adaptive", float (target positive ratio 0.0-1.0), or None
    )

    # Stopping criteria configuration (within-epoch)
    stopping_criterion_type: str = "plateau"
    patience: int = 20
    min_delta: float = 0.01
    accuracy_floor: float | None = (
        None  # Min accuracy before patience counts; None = auto-calc from data
    )

    # Cross-cycle stopping criteria (for active learning loops)
    cycle_stopping_strategy: str = "none"  # 'label', 'search', or 'none'
    cycle_patience: int = 5  # Cycles without improvement before stopping
    cycle_min_delta: float = 0.02  # Min improvement to reset patience
    cycle_min_cycles: int = 10  # Min cycles before stopping allowed
    cycle_window: int = 3  # Cycles to average for rolling metrics
    cycle_std_threshold: float = 0.08  # Max std dev for "stable" performance
    precision_floor: float | str = "auto"  # Min precision for 'search' mode; "auto" or 0.0-1.0

    # Active learning parameters (experiment configuration)
    total_candidates: int = 50
    positive_percentage: float | None = None  # None = no stratification (pure entropy)
    min_confidence: float = 0.8
    selection_mode: str = "entropy"
    candidate_pool_multiplier: int = 5  # Select from pool of N*total_candidates before stratifying

    # Selection strategy configuration (for 'basic_transition' mode)
    basic_transition_f1_threshold: float = 0.2  # F1 threshold to trigger strategy switch
    basic_transition_confidence_threshold: float = 0.9  # Confidence threshold to trigger switch
    basic_transition_variance_threshold: float = 0.12  # Variance threshold to trigger switch
    auto_thresholds: bool = (
        False  # Auto-calculate thresholds from dataset (overrides manual values)
    )
    estimated_positive_pct: float | None = (
        None  # Estimated positive % for auto-threshold calculation
    )

    # Ground truth evaluation configuration
    with_ground_truth: bool = (
        False  # Evaluation mode: add ground_truth/correct columns to predictions CSV
    )

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
        # Note: stopping_criterion_type validation happens in create_stopping_criterion()
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
        if self.candidate_pool_multiplier < 1:
            raise ValueError("candidate_pool_multiplier must be at least 1")
        # Note: cycle_stopping_strategy validation happens in create_cycle_stopping_criterion()

        # Validate precision_floor
        if isinstance(self.precision_floor, str):
            if self.precision_floor != "auto":
                raise ValueError("precision_floor must be 'auto' or a float between 0.0 and 1.0")
        elif isinstance(self.precision_floor, int | float):
            if not (0.0 <= self.precision_floor <= 1.0):
                raise ValueError("precision_floor must be between 0.0 and 1.0")
        else:
            raise ValueError("precision_floor must be 'auto' or a float between 0.0 and 1.0")

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

    @classmethod
    def from_yaml(cls, yaml_path: str | Path, **overrides) -> "AudioLoopConfig":
        """
        Load AudioLoopConfig from YAML file with optional overrides.

        Args:
            yaml_path: Path to YAML config file
            **overrides: Additional parameter overrides (highest precedence)

        Returns:
            AudioLoopConfig instance with merged configuration

        Raises:
            FileNotFoundError: If yaml_path doesn't exist
            yaml.YAMLError: If YAML is malformed
            ValueError: If YAML contains invalid field names or values

        Precedence (highest to lowest):
            1. overrides (explicit kwargs)
            2. YAML file values
            3. Environment variables (via default_factory)
            4. Dataclass defaults

        Example:
            >>> config = AudioLoopConfig.from_yaml("configs/experiment.yaml")
            >>> config = AudioLoopConfig.from_yaml(
            ...     "configs/base.yaml",
            ...     max_epochs=500,  # Override YAML value
            ...     batch_size=64
            ... )

        Supported YAML formats:
            Two-section format (recommended):
                workflow:
                    class_name: "Dog"
                    num_cycles: 10
                config:
                    experiment_name: "test"
                    dataset: "fsd50k"
                    max_epochs: 500

            Flat format (alternative):
                experiment_name: "test"
                dataset: "fsd50k"
                max_epochs: 500
        """
        import dataclasses

        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        # Load YAML
        try:
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {yaml_path}: {e}") from e

        if not yaml_data:
            yaml_data = {}

        # Extract 'config' section if present (two-section format)
        # Also support flat format (all fields at root level)
        if "config" in yaml_data:
            config_params = yaml_data["config"] or {}
        else:
            # Flat format - use entire YAML, but exclude 'workflow' section if present
            config_params = {k: v for k, v in yaml_data.items() if k != "workflow"}

        # Validate field names
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        invalid_fields = set(config_params.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"Invalid config fields in {yaml_path}: {', '.join(sorted(invalid_fields))}\n"
                f"Valid fields: {', '.join(sorted(valid_fields))}"
            )

        # Handle Path conversion for subset_csv
        if "subset_csv" in config_params and config_params["subset_csv"] is not None:
            config_params["subset_csv"] = Path(config_params["subset_csv"])

        # Merge: overrides > YAML > defaults (None values filtered from overrides)
        merged_params = {**config_params}
        merged_params.update({k: v for k, v in overrides.items() if v is not None})

        # Create config (will use env vars for any fields not in merged_params)
        return cls(**merged_params)

    @classmethod
    def from_project(cls, **overrides) -> "AudioLoopConfig":
        """
        Load AudioLoopConfig with project defaults from audioloop.yaml.

        Finds the project root (directory containing audioloop.yaml) and loads
        any settings from that file as defaults. Explicit overrides take precedence.

        Args:
            **overrides: Parameter overrides (highest precedence)

        Returns:
            AudioLoopConfig instance with merged configuration

        Raises:
            RuntimeError: If no project found (no audioloop.yaml in cwd and
                         AUDIOLOOP_PROJECT_ROOT not set)

        Precedence (highest to lowest):
            1. overrides (explicit kwargs)
            2. audioloop.yaml values (project defaults)
            3. Environment variables (via default_factory)
            4. Dataclass defaults

        Example:
            >>> # Load with project defaults
            >>> config = AudioLoopConfig.from_project()

            >>> # Load with project defaults, but override batch_size
            >>> config = AudioLoopConfig.from_project(batch_size=64)
        """
        from .utils.paths import get_project_root

        project_root = get_project_root()
        project_yaml = project_root / "audioloop.yaml"

        if project_yaml.exists():
            return cls.from_yaml(project_yaml, **overrides)

        # Project root exists (via env var) but no yaml file
        # Just use overrides + defaults
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
        return cls(**filtered_overrides)

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
        if self.specs_dir_path is not None:
            p = Path(self.specs_dir_path)
            return p if p.is_absolute() else get_project_root() / p
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
        from .datasets.dataset_registry import create_dataset

        return create_dataset(self.dataset, subset_csv=self.subset_csv)

    def get_feature_extractor(self, dataset_config: DatasetConfig | None = None):
        """Build the feature extractor for this experiment, with params from config.

        This is the config-scoped construction point: feature_extractor_type selects the
        extractor class and feature_extractor_kwargs (sample_rate, n_fft, n_mels, ...) come
        from the experiment config, so offline (create_specs) and lazy (SpectrogramDataset)
        builds share identical params. Pass an existing dataset_config to reuse it;
        otherwise one is built from this config.

        Selection is a small explicit dict rather than file-discovery (as models use): the
        set of extractors is tiny and the boring dict is the correct minimum. An unknown
        type fails here, the dispatch point, mirroring model_type/stopping_criterion_type.
        """
        from .feature_extractor import EmbeddingExtractor, SpectrogramExtractor

        extractor_classes = {
            "spectrogram": SpectrogramExtractor,
            "embedding": EmbeddingExtractor,
        }
        try:
            extractor_class = extractor_classes[self.feature_extractor_type]
        except KeyError:
            raise ValueError(
                f"Unknown feature_extractor_type {self.feature_extractor_type!r}; "
                f"expected one of {sorted(extractor_classes)}"
            ) from None

        if dataset_config is None:
            dataset_config = self.get_dataset_config()
        return extractor_class(dataset_config, **self.feature_extractor_kwargs)

    def create_directories(self) -> None:
        """Create all necessary directories for this configuration."""
        create_output_directories(self.experiment_name)
