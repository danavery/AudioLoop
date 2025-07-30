import argparse
import os
import random

from audioloop.config import AudioLoopConfig


def get_matching_samples(
    dataset_name: str, class_name: str | None = None, invert: bool = False, **kwargs
) -> list[str]:
    """Get matching samples for any supported dataset.

    Args:
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        class_name: Class name to filter by (if None, returns all)
        invert: If True, return samples NOT matching the class
        **kwargs: Additional dataset-specific parameters

    Returns:
        List of spectrogram filenames (not full paths)
    """
    config = AudioLoopConfig(dataset=dataset_name)
    dataset_config = config.get_dataset_config()

    # Get split parameter (FSD50K uses it, UrbanSound8K ignores it)
    split = kwargs.get("split", "dev")
    metadata = dataset_config.load_metadata(split=split)

    # Get positive class ID if we have a class name
    positive_class_id = None
    if class_name is not None:
        try:
            positive_class_id = dataset_config.name_to_id[class_name]
        except KeyError:
            raise ValueError(
                f"Invalid class name: '{class_name}'. Valid names: {list(dataset_config.name_to_id.keys())}"
            ) from None

    matching_filenames = []

    for item in metadata:
        # Determine if this sample matches our criteria
        if class_name is None:
            match = True
        else:
            # positive_class_id is guaranteed to be int when class_name is not None
            assert positive_class_id is not None
            is_positive = dataset_config.get_binary_label(item, positive_class_id, class_name)
            match = bool(is_positive)

        if invert:
            match = not match

        if match:
            # Check if the audio file actually exists before adding to candidates
            audio_path = item.get("audio_path") or dataset_config.get_audio_path(item["filename"])
            if audio_path.exists():
                # Use config's method to get spectrogram path
                spec_path = dataset_config.get_spectrogram_path(item["filename"], config.specs_dir)
                matching_filenames.append(spec_path.name)

    return matching_filenames


def write_starting_labels(
    n_positive: int = 10,
    n_negative: int = 10,
    class_name: str = "siren",
    dataset_name: str = "urbansound8k",
    **kwargs,
) -> tuple[list[str], list[str]]:
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
    positive_candidates = get_matching_samples(
        dataset_name, class_name=class_name, invert=False, **kwargs
    )
    if len(positive_candidates) < n_positive:
        raise ValueError(
            f"Not enough positive samples for class '{class_name}'. "
            f"Requested: {n_positive}, Available: {len(positive_candidates)}"
        )
    positives = random.sample(positive_candidates, k=n_positive)

    # Get negative samples
    negative_candidates = get_matching_samples(
        dataset_name, class_name=class_name, invert=True, **kwargs
    )
    if len(negative_candidates) < n_negative:
        raise ValueError(
            f"Not enough negative samples for class '{class_name}'. "
            f"Requested: {n_negative}, Available: {len(negative_candidates)}"
        )
    negatives = random.sample(negative_candidates, k=n_negative)

    return positives, negatives


def create_training_set(
    n: int = 10,
    class_name: str = "siren",
    dataset_name: str = "urbansound8k",
    output_path: str | None = None,
    positive_percentage: float = 0.5,
    experiment_name: str | None = None,
    **kwargs,
) -> tuple[list[str], list[str]]:
    """Create training set CSV file for any dataset and class.

    Args:
        n: Total number of samples
        class_name: Class name for positive class
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        output_path: Path to save training set CSV
        positive_percentage: Percentage of samples that should be positive (0.0-1.0)
        experiment_name: Optional experiment name to adjust output path
        **kwargs: Additional dataset-specific parameters
    """
    # Generate output path using config if not provided
    if output_path is None:
        from audioloop.config import AudioLoopConfig

        config = AudioLoopConfig(experiment_name=experiment_name)
        output_path = str(config.get_training_set_path(1))

    n_positive = int(n * positive_percentage)
    n_negative = n - n_positive

    positives, negatives = write_starting_labels(
        n_positive, n_negative, class_name, dataset_name, **kwargs
    )

    # Format entries
    positive_entries = [f"{p},1" for p in positives]
    negative_entries = [f"{n},0" for n in negatives]
    all_entries = positive_entries + negative_entries

    # Shuffle and write
    random.shuffle(all_entries)

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("filepath,label\n")
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
    config = AudioLoopConfig(dataset=dataset_name)
    dataset_config = config.get_dataset_config()

    # All datasets support list_classes()
    dataset_config.list_classes()

    # FSD50K has additional semantic groups information
    if dataset_name == "fsd50k":
        try:
            from audioloop.datasets.fsd50k_config import list_semantic_groups

            print("\n")
            list_semantic_groups()
        except ImportError:
            # If semantic groups function not available, just continue
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create initial training sets for any supported dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create FSD50K training set (default)
  python -m audioloop.utils.start_labeling --class-name Drill --n 50

  # Create UrbanSound8K training set
  python -m audioloop.utils.start_labeling --dataset urbansound8k --class-name siren --n 40

  # List available classes for FSD50K
  python -m audioloop.utils.start_labeling --list-classes

  # List available classes for UrbanSound8K
  python -m audioloop.utils.start_labeling --dataset urbansound8k --list-classes

  # Use custom paths
  python -m audioloop.utils.start_labeling --dataset urbansound8k --class-name dog_bark \\
    --metadata-csv /path/to/UrbanSound8K.csv --audio-root /path/to/audio
        """,
    )

    # Dataset selection - use dynamic discovery
    from audioloop.datasets.dataset_registry import list_available_datasets

    available_datasets = list_available_datasets()

    parser.add_argument(
        "--dataset",
        choices=available_datasets,
        default=None,
        help=f"Dataset to use. Available: {', '.join(available_datasets)}. Can also be set via AUDIOLOOP_DATASET environment variable.",
    )

    # Core parameters
    parser.add_argument("--class-name", default="siren", help="Class name for positive samples")
    parser.add_argument("--n", type=int, default=40, help="Total number of samples")
    parser.add_argument(
        "--positive-pct",
        type=float,
        default=0.75,
        help="Percentage positive (0.0-1.0, default 0.75)",
    )
    parser.add_argument(
        "--output", default=None, help="Output path (default: auto-generated from config)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to customize training set directory (default: training_sets, with experiment: training_sets_experiment)",
    )

    # Dataset-specific parameters
    parser.add_argument("--metadata-csv", help="Path to metadata CSV (UrbanSound8K)")
    parser.add_argument("--audio-root", help="Path to audio root directory")
    parser.add_argument("--output-dir", help="Path to spectrogram output directory")
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "eval"],
        help="Dataset split for FSD50K (default: dev)",
    )

    # Utility options
    parser.add_argument(
        "--list-classes", action="store_true", help="List all available classes for the dataset"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Resolve dataset choice from CLI and environment variable
    try:
        config = AudioLoopConfig(dataset=args.dataset)
        dataset_name = config.dataset
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # Set random seed if provided, otherwise use default
    if args.seed is not None:
        random.seed(args.seed)
    else:
        random.seed(42)  # Default seed for reproducibility

    # Build kwargs for dataset processor
    dataset_kwargs = {}
    if args.metadata_csv:
        dataset_kwargs["metadata_csv"] = args.metadata_csv
    if args.audio_root:
        dataset_kwargs["audio_root"] = args.audio_root
    if args.output_dir:
        dataset_kwargs["output_dir"] = args.output_dir
    if dataset_name == "fsd50k":
        dataset_kwargs["split"] = args.split

    if args.list_classes:
        list_available_classes(dataset_name, **dataset_kwargs)
    else:
        try:
            positive_files, negative_files = create_training_set(
                n=args.n,
                class_name=args.class_name,
                dataset_name=dataset_name,
                output_path=args.output,
                positive_percentage=args.positive_pct,
                experiment_name=args.experiment,
                **dataset_kwargs,
            )
        except ValueError as e:
            print(f"Error: {e}")
            if "Not enough" in str(e):
                print("\nTip: Use --list-classes to see available classes, or reduce --n")
            exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            exit(1)
