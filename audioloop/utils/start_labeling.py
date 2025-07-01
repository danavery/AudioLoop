import argparse
import os
import random
from pathlib import Path

from audioloop.datasets.fsd50k import FSD50KConfig, FSD50KProcessor
from audioloop.datasets.urbansound8k import UrbanSound8KConfig, UrbanSound8KProcessor
from audioloop.utils.dataset_utils import get_dataset_help_text, resolve_dataset_choice


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
        config = UrbanSound8KConfig()
        # Override config paths if provided
        if 'metadata_csv' in kwargs:
            config.metadata_csv = Path(kwargs['metadata_csv'])
        if 'audio_root' in kwargs:
            config.audio_root = Path(kwargs['audio_root'])
        if 'output_dir' in kwargs:
            config.output_dir = Path(kwargs['output_dir'])
        processor = UrbanSound8KProcessor(config)
        return processor, config
    if dataset_name == "fsd50k":
        config = FSD50KConfig()
        # Override config paths if provided
        if 'metadata_dir' in kwargs:
            config.metadata_dir = Path(kwargs['metadata_dir'])
        if 'ground_truth_dir' in kwargs:
            config.ground_truth_dir = Path(kwargs['ground_truth_dir'])
        if 'audio_root' in kwargs:
            config.audio_root = Path(kwargs['audio_root'])
        if 'output_dir' in kwargs:
            config.output_dir = Path(kwargs['output_dir'])
        processor = FSD50KProcessor(config)
        return processor, config
    raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: urbansound8k, fsd50k")


def get_matching_samples_urbansound8k(processor, config, class_name: str | None = None, invert: bool = False) -> list[str]:
    """Get matching samples for UrbanSound8K dataset.

    Args:
        processor: UrbanSound8K processor instance
        config: UrbanSound8K config instance
        class_name: Class name to filter by (if None, returns all)
        invert: If True, return samples NOT matching the class

    Returns:
        List of spectrogram file paths
    """
    metadata = processor.load_metadata()
    matching_paths = []

    for item in metadata:
        filename = item["filename"]
        original_class = item["class_name"]

        # Determine if this sample matches our criteria
        match = True if class_name is None else original_class == class_name

        if invert:
            match = not match

        if match:
            # Convert to spectrogram path
            spec_filename = filename.replace('.wav', '.pt')
            spec_path = config.output_dir / spec_filename
            matching_paths.append(str(spec_path))

    return matching_paths


def get_matching_samples_fsd50k(processor, config, class_name: str | None = None, invert: bool = False, split: str = "dev") -> list[str]:
    """Get matching samples for FSD50K dataset.

    Args:
        processor: FSD50K processor instance
        config: FSD50K config instance
        class_name: Class name to filter by (if None, returns all)
        invert: If True, return samples NOT matching the class
        split: Dataset split to use ('dev' or 'eval')

    Returns:
        List of spectrogram file paths
    """
    metadata = processor.load_metadata(split=split)
    matching_paths = []

    for item in metadata:
        filename = item["filename"]
        labels = item["labels"]

        # Determine if this sample matches our criteria
        match = True if class_name is None else class_name in labels

        if invert:
            match = not match

        if match:
            # Convert to spectrogram path
            spec_filename = f"{filename}.pt"
            spec_path = config.output_dir / spec_filename
            matching_paths.append(str(spec_path))

    return matching_paths


def get_matching_samples(dataset_name: str, class_name: str | None = None, invert: bool = False, **kwargs) -> list[str]:
    """Get matching samples for any supported dataset.

    Args:
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        class_name: Class name to filter by (if None, returns all)
        invert: If True, return samples NOT matching the class
        **kwargs: Additional dataset-specific parameters

    Returns:
        List of spectrogram file paths
    """
    processor, config = get_dataset_processor(dataset_name, **kwargs)

    if dataset_name == "urbansound8k":
        return get_matching_samples_urbansound8k(processor, config, class_name, invert)
    if dataset_name == "fsd50k":
        split = kwargs.get('split', 'dev')
        return get_matching_samples_fsd50k(processor, config, class_name, invert, split)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def write_starting_labels(n_positive: int = 10, n_negative: int = 10,
                         class_name: str = "siren", dataset_name: str = "urbansound8k",
                         **kwargs) -> tuple[list[str], list[str]]:
    """Create initial training set for any dataset and class.

    Args:
        n_positive: Number of positive samples
        n_negative: Number of negative samples
        class_name: Class name for positive class
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        **kwargs: Additional dataset-specific parameters

    Returns:
        (positives, negatives): Lists of file paths
    """
    # Get positive samples
    positive_candidates = get_matching_samples(dataset_name, class_name=class_name, invert=False, **kwargs)
    if len(positive_candidates) < n_positive:
        raise ValueError(f"Not enough positive samples for class '{class_name}'. "
                        f"Requested: {n_positive}, Available: {len(positive_candidates)}")
    positives = random.sample(positive_candidates, k=n_positive)

    # Get negative samples
    negative_candidates = get_matching_samples(dataset_name, class_name=class_name, invert=True, **kwargs)
    if len(negative_candidates) < n_negative:
        raise ValueError(f"Not enough negative samples for class '{class_name}'. "
                        f"Requested: {n_negative}, Available: {len(negative_candidates)}")
    negatives = random.sample(negative_candidates, k=n_negative)

    return positives, negatives


def create_training_set(
    n: int = 10,
    class_name: str = "siren",
    dataset_name: str = "urbansound8k",
    output_path: str = "training_sets/training_set_v1.csv",
    run: int = 1,
    positive_percentage: float = 0.5,
    **kwargs
) -> tuple[list[str], list[str]]:
    """Create training set CSV file for any dataset and class.

    Args:
        n: Total number of samples
        class_name: Class name for positive class
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        output_path: Path to save training set CSV
        run: Run number for training set versioning
        positive_percentage: Percentage of samples that should be positive (0.0-1.0)
        **kwargs: Additional dataset-specific parameters
    """
    n_positive = int(n * positive_percentage)
    n_negative = n - n_positive

    positives, negatives = write_starting_labels(
        n_positive, n_negative, class_name, dataset_name, **kwargs
    )

    # Format entries
    positive_entries = [f"{p},1,{run}" for p in positives]
    negative_entries = [f"{n},0,{run}" for n in negatives]
    all_entries = positive_entries + negative_entries

    # Shuffle and write
    random.shuffle(all_entries)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("filepath,label,run\n")
        f.write("\n".join(all_entries))

    print(f"Created training set: {output_path}")
    print(f"  Dataset: {dataset_name}")
    print(f"  {len(positives)} {class_name} samples, {len(negatives)} non-{class_name} samples")
    print(f"  Positive percentage: {len(positives) / (len(positives) + len(negatives)):.1%}")

    return positives, negatives


def list_available_classes(dataset_name: str, **kwargs) -> None:
    """List all available classes for a dataset.

    Args:
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        **kwargs: Additional dataset-specific parameters
    """
    processor, config = get_dataset_processor(dataset_name, **kwargs)

    if dataset_name == "urbansound8k":
        from audioloop.datasets.urbansound8k import list_classes
        list_classes()
    elif dataset_name == "fsd50k":
        from audioloop.datasets.fsd50k import list_classes, list_semantic_groups
        vocab_path = kwargs.get('vocab_path', getattr(config, 'vocabulary_csv', None))
        list_classes(vocab_path)
        print("\n")
        list_semantic_groups()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create initial training sets for any supported dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create UrbanSound8K training set (default)
  python -m audioloop.utils.start_labeling --class-name siren --n 40

  # Create FSD50K training set
  python -m audioloop.utils.start_labeling --dataset fsd50k --class-name Drill --n 50

  # List available classes for UrbanSound8K
  python -m audioloop.utils.start_labeling --list-classes

  # List available classes for FSD50K
  python -m audioloop.utils.start_labeling --dataset fsd50k --list-classes

  # Use custom paths
  python -m audioloop.utils.start_labeling --dataset urbansound8k --class-name dog_bark \\
    --metadata-csv /path/to/UrbanSound8K.csv --audio-root /path/to/audio
        """
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=["urbansound8k", "fsd50k"],
        default=None,
        help=get_dataset_help_text()
    )

    # Core parameters
    parser.add_argument("--class-name", default="siren", help="Class name for positive samples")
    parser.add_argument("--n", type=int, default=40, help="Total number of samples")
    parser.add_argument(
        "--positive-pct",
        type=float,
        default=0.75,
        help="Percentage positive (0.0-1.0, default 0.75)"
    )
    parser.add_argument("--output", default="training_sets/training_set_v1.csv", help="Output path")
    parser.add_argument("--run", type=int, default=1, help="Run number for versioning")

    # Dataset-specific parameters
    parser.add_argument("--metadata-csv", help="Path to metadata CSV (UrbanSound8K)")
    parser.add_argument("--audio-root", help="Path to audio root directory")
    parser.add_argument("--output-dir", help="Path to spectrogram output directory")
    parser.add_argument("--split", default="dev", choices=["dev", "eval"],
                       help="Dataset split for FSD50K (default: dev)")

    # Utility options
    parser.add_argument("--list-classes", action="store_true",
                       help="List all available classes for the dataset")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Resolve dataset choice from CLI and environment variable
    try:
        dataset_name = resolve_dataset_choice(args.dataset)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Build kwargs for dataset processor
    dataset_kwargs = {}
    if args.metadata_csv:
        dataset_kwargs['metadata_csv'] = args.metadata_csv
    if args.audio_root:
        dataset_kwargs['audio_root'] = args.audio_root
    if args.output_dir:
        dataset_kwargs['output_dir'] = args.output_dir
    if dataset_name == "fsd50k":
        dataset_kwargs['split'] = args.split

    if args.list_classes:
        list_available_classes(dataset_name, **dataset_kwargs)
    else:
        try:
            create_training_set(
                n=args.n,
                class_name=args.class_name,
                dataset_name=dataset_name,
                output_path=args.output,
                run=args.run,
                positive_percentage=args.positive_pct,
                **dataset_kwargs
            )
        except ValueError as e:
            print(f"Error: {e}")
            if "Not enough" in str(e):
                print("\nTip: Use --list-classes to see available classes, or reduce --n")
            exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            exit(1)
