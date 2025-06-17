#!/usr/bin/env python3
"""
Example script demonstrating how to use the generalized active learning framework.

This script shows how to run active learning cycles for different binary classification
tasks using the UrbanSound8K dataset.
"""

import argparse
import sys
from pathlib import Path

from .active_learning import run_active_learning_cycle
from .urbansound_classes import (
    URBANSOUND8K_CLASSES,
    create_negative_class_name,
    get_binary_preset,
    get_class_name,
    list_binary_presets,
)


def run_preset_cycle(preset_name, model_path, cycle_name=None):
    """Run an active learning cycle using a predefined preset.

    Args:
        preset_name (str): Name of the binary classification preset
        model_path (str): Path to trained model
        cycle_name (str, optional): Name for this cycle

    Returns:
        tuple: (predictions_file, candidates_file)
    """
    config = get_binary_preset(preset_name)

    if cycle_name is None:
        cycle_name = f"{preset_name}_cycle1"

    print(f"Running active learning cycle: {preset_name}")
    print(f"Positive class: {config['positive_class_name']} (ID: {config['positive_class_id']})")
    print(f"Negative class: {config['negative_class_name']}")
    print(f"Model: {model_path}")
    print(f"Cycle name: {cycle_name}")
    print("-" * 60)

    return run_active_learning_cycle(
        positive_class_id=config['positive_class_id'],
        positive_class_name=config['positive_class_name'],
        negative_class_name=config['negative_class_name'],
        model_path=model_path,
        cycle_name=cycle_name
    )


def run_custom_cycle(class_id, model_path, cycle_name=None, positive_name=None, negative_name=None):
    """Run an active learning cycle for a custom class.

    Args:
        class_id (int): UrbanSound8K class ID to use as positive class
        model_path (str): Path to trained model
        cycle_name (str, optional): Name for this cycle
        positive_name (str, optional): Custom name for positive class
        negative_name (str, optional): Custom name for negative class

    Returns:
        tuple: (predictions_file, candidates_file)
    """
    # Get default class name if not provided
    if positive_name is None:
        positive_name = get_class_name(class_id)

    if negative_name is None:
        negative_name = create_negative_class_name(positive_name)

    if cycle_name is None:
        cycle_name = f"{positive_name}_cycle1"

    print("Running custom active learning cycle")
    print(f"Positive class: {positive_name} (ID: {class_id})")
    print(f"Negative class: {negative_name}")
    print(f"Model: {model_path}")
    print(f"Cycle name: {cycle_name}")
    print("-" * 60)

    return run_active_learning_cycle(
        positive_class_id=class_id,
        positive_class_name=positive_name,
        negative_class_name=negative_name,
        model_path=model_path,
        cycle_name=cycle_name
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run active learning cycles for binary audio classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run siren detection with preset
  python -m audioloop.run_active_learning --preset siren_detection --model outputs/model_v1.pt

  # Run custom dog bark detection
  python -m audioloop.run_active_learning --class-id 3 --model outputs/model_v1.pt

  # Run gunshot detection with custom names
  python -m audioloop.run_active_learning --class-id 6 --model outputs/model_v1.pt \\
    --positive-name "gunshot" --negative-name "safe_sound" --cycle-name "security_v1"

  # List available presets
  python -m audioloop.run_active_learning --list-presets
        """
    )

    # Main mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--preset', type=str,
                           help='Use a predefined binary classification preset')
    mode_group.add_argument('--class-id', type=int, choices=range(10),
                           help='UrbanSound8K class ID to use as positive class (0-9)')
    mode_group.add_argument('--list-presets', action='store_true',
                           help='List available binary classification presets')
    mode_group.add_argument('--list-classes', action='store_true',
                           help='List all UrbanSound8K classes')

    # Required for actual runs
    parser.add_argument('--model', type=str,
                       help='Path to trained model file (.pt)')

    # Optional customization
    parser.add_argument('--cycle-name', type=str,
                       help='Name for this active learning cycle (used in output filenames)')
    parser.add_argument('--positive-name', type=str,
                       help='Custom name for positive class')
    parser.add_argument('--negative-name', type=str,
                       help='Custom name for negative class')

    # Parse arguments
    args = parser.parse_args()

    # Handle list commands
    if args.list_presets:
        list_binary_presets()
        return

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

    predictions_file = None
    candidates_file = None
    try:
        # Run the appropriate cycle
        if args.preset:
            predictions_file, candidates_file = run_preset_cycle(
                preset_name=args.preset,
                model_path=str(model_path),
                cycle_name=args.cycle_name
            )

        elif args.class_id is not None:
            predictions_file, candidates_file = run_custom_cycle(
                class_id=args.class_id,
                model_path=str(model_path),
                cycle_name=args.cycle_name,
                positive_name=args.positive_name,
                negative_name=args.negative_name
            )

        print("\n✅ Active learning cycle completed successfully!")
        print(f"📊 Predictions: {predictions_file}")
        print(f"🏷️  Candidates: {candidates_file}")

    except Exception as e:
        print(f"❌ Error running active learning cycle: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
