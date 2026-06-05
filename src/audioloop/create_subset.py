#!/usr/bin/env python3
"""
Create dataset subsets for binary classification.

Generates training-ready CSV files with format:
    filename,label,original_class,split,audio_path

Usage:
    python -m audioloop.create_subset --dataset audioset --class-name "Speech" --max-samples 50000
    python -m audioloop.create_subset --dataset audioset --class-name "Brass instrument" --max-samples 100000 --positive-ratio 0.05
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset subset for binary classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 50k sample subset with 50% positive ratio (default)
  python -m audioloop.create_subset --dataset audioset --class-name "Speech" --max-samples 50000

  # Create 100k sample subset with 5% positive ratio from unbalanced train
  python -m audioloop.create_subset --dataset audioset --class-name "Brass instrument" \\
      --max-samples 100000 --positive-ratio 0.05 --split unbal_train

  # Custom output location
  python -m audioloop.create_subset --dataset audioset --class-name "Dog" \\
      --max-samples 10000 --output my_subsets/dog_subset.csv
        """,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["audioset"],
        help="Dataset to create subset from (currently only audioset supported)",
    )
    parser.add_argument(
        "--class-name",
        help="Class name to use as positive class (e.g., 'Speech', 'Brass instrument')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum total samples (positive + negative combined)",
    )
    parser.add_argument(
        "--positive-ratio",
        type=float,
        default=0.5,
        help="Target ratio of positive samples (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to sample from (default: dataset-specific, usually largest split)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: subsets/DATASET_CLASSNAME_SAMPLES.csv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="List all available classes for the dataset and exit",
    )

    args = parser.parse_args()

    # Import config after argparse to keep startup fast
    from audioloop.config import AudioLoopConfig

    # Get dataset config
    config = AudioLoopConfig.from_project(dataset=args.dataset)
    dataset_config = config.get_dataset_config()

    # Handle --list-classes
    if args.list_classes:
        dataset_config.list_classes()
        return 0

    # Validate required args when not listing classes
    if not args.class_name:
        parser.error("--class-name is required (unless using --list-classes)")
    if not args.max_samples:
        parser.error("--max-samples is required (unless using --list-classes)")

    # Validate positive ratio
    if not (0 < args.positive_ratio < 1):
        parser.error("--positive-ratio must be between 0 and 1")

    # Generate default output path if not specified
    if args.output is None:
        safe_name = args.class_name.replace(" ", "_").lower()
        output_dir = Path("subsets")
        output_dir.mkdir(exist_ok=True)
        args.output = output_dir / f"{args.dataset}_{safe_name}_{args.max_samples}.csv"

    # Create subset
    print(f"Creating subset from {args.dataset}...")
    print(f"  Class: {args.class_name}")
    print(f"  Samples: {args.max_samples:,} ({args.positive_ratio:.1%} positive)")
    if args.split:
        print(f"  Split: {args.split}")

    try:
        result_path = dataset_config.create_subset(
            output_path=args.output,
            class_name=args.class_name,
            max_samples=args.max_samples,
            positive_ratio=args.positive_ratio,
            split=args.split,
            seed=args.seed,
        )
        print(f"\n✅ Subset saved to: {result_path}")
        print("\nNext steps:")
        print("  1. Generate specs (optional, can be done lazily during training):")
        print(
            f"     python -m audioloop.create_specs --dataset {args.dataset} --metadata-file {result_path}"
        )
        print("\n  2. Create initial training set:")
        print("     # For evaluation mode (with ground truth):")
        print(
            f'     python -m audioloop.utils.create_bootstrap_set --class-name "{args.class_name}" \\'
        )
        print(f"       --dataset {args.dataset} --subset-csv {result_path} --n 40 \\")
        print("       --experiment my_experiment")
        print(
            "     # For production mode: manually label ~40 samples and create training_set_v1.csv"
        )
        print("\n  3. Run automated active learning workflow:")
        print("     python -m audioloop.automated_workflow --experiment-name my_experiment \\")
        print(f'       --dataset {args.dataset} --class-name "{args.class_name}" \\')
        print(f"       --subset-csv {result_path} --cycles 5 --evaluation-mode --auto-label")
        print("\n  Or manually train a model:")
        print("     python -m audioloop.train training_sets/training_set_v1.csv")
        return 0

    except NotImplementedError as e:
        parser.error(str(e))
    except ValueError as e:
        parser.error(f"Invalid class name: {e}")


if __name__ == "__main__":
    sys.exit(main())
