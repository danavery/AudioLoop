import os
from pathlib import Path
from typing import Literal

DatasetType = Literal["urbansound8k", "fsd50k"]

DEFAULT_DATASET: DatasetType = "urbansound8k"
SUPPORTED_DATASETS = ["urbansound8k", "fsd50k"]


def get_default_dataset() -> DatasetType:
    """Get the default dataset from environment variable or fallback.

    Returns:
        Dataset name, either from AUDIOLOOP_DATASET environment variable
        or the default 'urbansound8k'

    Raises:
        ValueError: If AUDIOLOOP_DATASET is set to an unsupported value
    """
    env_dataset = os.environ.get("AUDIOLOOP_DATASET")

    if env_dataset is None:
        return DEFAULT_DATASET

    env_dataset = env_dataset.lower()
    if env_dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Invalid AUDIOLOOP_DATASET='{env_dataset}'. "
            f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
        )

    return env_dataset  # type: ignore


def resolve_dataset_choice(cli_dataset: str | None = None) -> DatasetType:
    """Resolve dataset choice from CLI argument and environment variable.

    Args:
        cli_dataset: Dataset specified via CLI argument (takes precedence)

    Returns:
        Resolved dataset name

    Raises:
        ValueError: If resolved dataset is not supported
    """
    if cli_dataset is not None:
        if cli_dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Invalid dataset choice: '{cli_dataset}'. "
                f"Supported datasets: {', '.join(SUPPORTED_DATASETS)}"
            )
        return cli_dataset  # type: ignore

    return get_default_dataset()


def get_dataset_processor(dataset_name: str, **kwargs):
    """Get the appropriate dataset processor and config.

    Args:
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        **kwargs: Additional configuration parameters for the dataset

    Returns:
        Tuple of (processor, config)

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name == "urbansound8k":
        from audioloop.datasets.urbansound8k import UrbanSound8KConfig, UrbanSound8KProcessor

        config = UrbanSound8KConfig()
        # Override config paths if provided
        if "metadata_csv" in kwargs:
            config.metadata_csv = Path(kwargs["metadata_csv"])
        if "audio_root" in kwargs:
            config.audio_root = Path(kwargs["audio_root"])
        if "output_dir" in kwargs:
            config.output_dir = Path(kwargs["output_dir"])
        processor = UrbanSound8KProcessor(config)
        return processor, config

    if dataset_name == "fsd50k":
        from audioloop.datasets.fsd50k import FSD50KConfig, FSD50KProcessor

        config = FSD50KConfig()
        # Override config paths if provided
        if "metadata_dir" in kwargs:
            config.metadata_dir = Path(kwargs["metadata_dir"])
        if "ground_truth_dir" in kwargs:
            config.ground_truth_dir = Path(kwargs["ground_truth_dir"])
        if "audio_root" in kwargs:
            config.audio_root = Path(kwargs["audio_root"])
        if "output_dir" in kwargs:
            config.output_dir = Path(kwargs["output_dir"])
        processor = FSD50KProcessor(config)
        return processor, config

    raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: urbansound8k, fsd50k")


def get_dataset_help_text() -> str:
    """Get help text for dataset argument that mentions environment variable."""
    env_dataset = os.environ.get("AUDIOLOOP_DATASET")
    if env_dataset:
        # Check if env var is valid before using it in help text
        try:
            resolved_dataset = get_default_dataset()
            return f"Dataset to use (default: {resolved_dataset} from AUDIOLOOP_DATASET)"
        except ValueError:
            # Invalid env var - show error message instead
            return (
                f"Dataset to use (default: {DEFAULT_DATASET}, AUDIOLOOP_DATASET has invalid value)"
            )
    else:
        return f"Dataset to use (default: {DEFAULT_DATASET}, or set AUDIOLOOP_DATASET)"
