import os
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
            return f"Dataset to use (default: {DEFAULT_DATASET}, AUDIOLOOP_DATASET has invalid value)"
    else:
        return f"Dataset to use (default: {DEFAULT_DATASET}, or set AUDIOLOOP_DATASET)"
