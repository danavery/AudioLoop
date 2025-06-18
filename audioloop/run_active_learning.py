#!/usr/bin/env python3
"""
Example script demonstrating how to use the generalized active learning framework.

This script shows how to run active learning cycles for different binary classification
tasks using the UrbanSound8K dataset.
"""

import argparse
import sys
from pathlib import Path

from .active_learning import run_active_learning_for_class
from .urbansound_classes import (
    URBANSOUND8K_CLASSES,
    get_class_name,
)


def run_cycle(positive_class_name, model_path, run_number=1, negative_name=None):
    """Run an active learning cycle for any class.

    Args:
        positive_class_name (str): UrbanSound8K class name to use as positive class
        model_path (str): Path to trained model
        run_number (int): Version number for output files
        negative_name (str, optional): Custom name for negative class

    Returns:
        tuple: (predictions_file, candidates_file)
    """
    print(f"Running active learning cycle: {positive_class_name} detection")
    print("-" * 60)
    print(f"Positive class: {positive_class_name}")
    print(f"Negative class: {negative_name or f'not_{positive_class_name}'}")
    print(f"Model: {model_path}")
    print(f"Run number: v{run_number}")
    print("-" * 60)

    return run_active_learning_for_class(
        positive_class_name=positive_class_name,
        model_path=model_path,
        negative_class_name=negative_name,
        run_number=run_number
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning cycles for binary audio classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dog bark detection
  python -m audioloop.run_active_learning --class-name dog_bark --model outputs/model_v1.pt

  # Run siren detection with class ID
  python -m audioloop.run_active_learning --class-id 8 --model outputs/model_v1.pt

  # Run gunshot detection with custom negative name
  python -m audioloop.run_active_learning --class-name gun_shot --model outputs/model_v1.pt \\
    --negative-name "safe_sound" --run-number 2

  # List all classes
  python -m audioloop.run_active_learning --list-classes
        """
    )

    # Main mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--class-name', type=str,
                           help='UrbanSound8K class name to use as positive class')
    mode_group.add_argument('--class-id', type=int, choices=range(10),
                           help='UrbanSound8K class ID to use as positive class (0-9)')
    mode_group.add_argument('--list-classes', action='store_true',
                           help='List all UrbanSound8K classes')

    # Required for actual runs
    parser.add_argument('--model', type=str,
                       help='Path to trained model file (.pt)')

    # Optional customization
    parser.add_argument('--run-number', type=int, default=1,
                       help='Version number for output files (default: 1)')
    parser.add_argument('--negative-name', type=str,
                       help='Custom name for negative class')

    # Parse arguments
    args = parser.parse_args()

    # Handle list commands
    if args.list_classes:
        print("UrbanSound8K Classes:")
        print("=" * 30)
        for class_id, name in URBANSOUND8K_CLASSES.items():
            print(f"{class_id}: {name}")
        return

    # Validate model path for actual runs
    if not args.model:
        print("Error: --model is required for running active learning cycles")
        parser.print_help()
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    try:
        # Determine class name
        positive_class_name = ""
        if args.class_name:
            positive_class_name = args.class_name
        elif args.class_id is not None:
            positive_class_name = get_class_name(args.class_id)

        # Run the cycle
        predictions_file, candidates_file = run_cycle(
            positive_class_name=positive_class_name,
            model_path=str(model_path),
            run_number=args.run_number,
            negative_name=args.negative_name
        )

        print("\n✅ Active learning cycle completed successfully!")
        print(f"📊 Predictions: {predictions_file}")
        print(f"🏷️  Candidates: {candidates_file}")
        print("\n📋 Next steps for human labeling:")
        print(f"1. Open {candidates_file}")
        print("2. Fill in the 'needs_human_label' column with 0 or 1")
        print("3. Merge labels back into training set:")
        print(f"   python -m audioloop.merge_labels training_sets/training_set_v{args.run_number}.csv {candidates_file}")

    except Exception as e:
        print(f"❌ Error running active learning cycle: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
