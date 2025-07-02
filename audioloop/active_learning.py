import argparse
import os
import re

from .active_learning_core import run_active_learning_cycle
from .utils.dataset_utils import (
    get_dataset_help_text,
    get_dataset_processor,
    resolve_dataset_choice,
)


def run_active_learning_for_class(
    positive_class_name,
    model_path,
    negative_class_name=None,
    dataset_name="urbansound8k",
    dataset_file=None,
    run_number=1,
    total_candidates=50,
    positive_percentage=0.75,
    min_confidence=0.8,
    **dataset_kwargs,
):
    """
    Simplified active learning cycle - just provide the class name.

    Args:
        positive_class_name: Audio class name (e.g., "dog_bark", "siren")
        model_path: Path to trained model
        negative_class_name: Name for negative class (auto-generated if None)
        dataset_name: Name of the dataset ('urbansound8k' or 'fsd50k')
        dataset_file: Path to dataset metadata CSV (auto-detected if None)
        run_number: Version number for output files
        total_candidates: Total number of candidates to select
        positive_percentage: Percentage of candidates that should be positive predictions (0.0-1.0)
        min_confidence: Minimum confidence threshold for candidate selection
        **dataset_kwargs: Additional dataset-specific configuration

    Returns:
        tuple: (predictions_file, candidates_file)
    """
    # Get processor and config once
    processor, config = get_dataset_processor(dataset_name, **dataset_kwargs)

    # Auto-detect dataset file if not provided
    if dataset_file is None:
        dataset_file = str(config.dataset_csv)

    # Validate class name and get class ID
    positive_class_id = processor.get_class_id(positive_class_name)

    # Auto-generate negative class name if not provided
    if negative_class_name is None:
        negative_class_name = f"not_{positive_class_name}"

    # Call the main function
    return run_active_learning_cycle(
        positive_class_id=positive_class_id,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_file=dataset_file,
        run_number=run_number,
        total_candidates=total_candidates,
        positive_percentage=positive_percentage,
        min_confidence=min_confidence,
    )


def main():
    """Main CLI entry point for active learning."""
    parser = argparse.ArgumentParser(
        description="Run active learning cycles for binary audio classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run siren detection cycle 1 (automatically uses outputs/model_v1.pt)
  python -m audioloop.active_learning --class-name siren --run-number 1

  # Run dog bark detection cycle 2 (automatically uses outputs/model_v2.pt)
  python -m audioloop.active_learning --class-name dog_bark --run-number 2

  # Run with FSD50K dataset
  python -m audioloop.active_learning --dataset fsd50k --class-name Drill --run-number 1

  # Run with class ID instead of name (UrbanSound8K only)
  python -m audioloop.active_learning --class-id 3 --run-number 1

  # Specify custom model path
  python -m audioloop.active_learning --class-name gun_shot --model custom_model.pt

  # Custom negative class name
  python -m audioloop.active_learning --class-name gun_shot --negative-name safe_sound --run-number 1

  # List all available classes
  python -m audioloop.active_learning --list-classes
        """,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        help=get_dataset_help_text(),
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--class-name",
        type=str,
        help="Audio class name to use as positive class (e.g., siren, dog_bark for UrbanSound8K; Drill, Dog for FSD50K)",
    )
    mode_group.add_argument(
        "--class-id",
        type=int,
        help="Audio class ID to use as positive class (UrbanSound8K: 0-9, FSD50K: varies)",
    )
    mode_group.add_argument(
        "--list-classes", action="store_true", help="List all available audio classes and exit"
    )

    # Required for actual runs
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model file (default: outputs/model_v{run_number}.pt)",
    )

    # Optional parameters
    parser.add_argument(
        "--run-number", type=int, default=1, help="Version number for output files (default: 1)"
    )
    parser.add_argument(
        "--negative-name",
        type=str,
        help="Custom name for negative class (default: not_<positive_class_name>)",
    )
    parser.add_argument(
        "--total-candidates",
        type=int,
        default=50,
        help="Total number of candidates to select (default: 20)",
    )
    parser.add_argument(
        "--positive-pct",
        type=float,
        default=0.75,
        help="Percentage of candidates that should be positive predictions (default: 0.75 for imbalanced)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence threshold for candidate selection (default: 0.8)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="Path to dataset metadata CSV file (auto-detected from dataset if not specified)",
    )

    args = parser.parse_args()

    # Resolve dataset choice
    try:
        dataset_name = resolve_dataset_choice(args.dataset)
    except ValueError as e:
        parser.error(str(e))

    # Get processor and config once for the entire operation
    processor, config = get_dataset_processor(dataset_name)

    # Handle list classes
    if args.list_classes:
        processor.list_classes()
        return

    # Default model path based on run_number if not specified
    if not args.model:
        args.model = f"outputs/model_v{args.run_number}.pt"
        print(f"Using default model path: {args.model}")
    else:
        # If model path provided, try to extract version number from it
        match = re.search(r"model_v(\d+)\.pt", args.model)
        if match and args.run_number == 1:  # Only override if using default run_number
            extracted_version = int(match.group(1))
            args.run_number = extracted_version
            print(f"Auto-detected run number {extracted_version} from model filename")

    if not os.path.exists(args.model):
        parser.error(f"Model file not found: {args.model}")

    # Determine class name and ID
    if args.class_name:
        positive_class_name = args.class_name
        try:
            positive_class_id = processor.get_class_id(args.class_name)
        except ValueError as e:
            parser.error(str(e))
    else:  # args.class_id is not None
        positive_class_id = args.class_id
        try:
            positive_class_name = processor.get_class_name(positive_class_id)
        except ValueError as e:
            parser.error(str(e))

    # Determine negative class name
    negative_class_name = args.negative_name or f"not_{positive_class_name}"

    # Print configuration
    print("Running active learning cycle")
    print("-" * 60)
    print(f"Positive class: {positive_class_name} (ID: {positive_class_id})")
    print(f"Negative class: {negative_class_name}")
    print(f"Model: {args.model}")
    print(f"Run number: {args.run_number}")
    num_positive = int(args.total_candidates * args.positive_pct)
    num_negative = args.total_candidates - num_positive
    print(
        f"Candidates: {num_positive} positive, {num_negative} negative ({args.positive_pct:.0%} positive)"
    )
    print(f"Min confidence: {args.min_confidence}")
    print("-" * 60)

    # Run the active learning cycle
    predictions_file, candidates_file = run_active_learning_cycle(
        positive_class_id=positive_class_id,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        model_path=args.model,
        dataset_name=dataset_name,
        dataset_file=args.dataset_file,
        run_number=args.run_number,
        total_candidates=args.total_candidates,
        positive_percentage=args.positive_pct,
        min_confidence=args.min_confidence,
    )

    print("\n✅ Active learning cycle completed!")
    print(f"📊 Predictions: {predictions_file}")
    print(f"🏷️  Candidates: {candidates_file}")
    print("\nNext steps:")
    print(f"1. Label candidates: python -m audioloop.label_audio {candidates_file}")
    print(
        f"2. Merge labels: python -m audioloop.merge_labels training_sets/training_set_v{args.run_number}.csv {candidates_file}"
    )


if __name__ == "__main__":
    main()
