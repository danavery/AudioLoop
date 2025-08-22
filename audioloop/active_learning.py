import argparse
import logging
import os
import re

from .active_learning_core import run_active_learning_cycle
from .config import AudioLoopConfig


def run_active_learning_for_class(
    positive_class_name,
    model_path,
    negative_class_name=None,
    dataset_name="urbansound8k",
    dataset_file=None,
    run_number=1,
    training_set_csv=None,
    total_candidates=50,
    positive_percentage=0.75,
    min_confidence=0.8,
    selection_mode="confidence",
    basic_transition_f1_threshold=0.2,
    basic_transition_confidence_threshold=0.9,
    basic_transition_variance_threshold=0.12,
    auto_thresholds=False,
    estimated_positive_pct=None,
    experiment_name=None,
    seed=None,
    with_ground_truth=False,
    log_level=logging.INFO,
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
        training_set_csv: Path to training set CSV (auto-detected if None)
        total_candidates: Total number of candidates to select
        positive_percentage: Percentage of candidates that should be positive predictions (0.0-1.0)
        min_confidence: Minimum confidence threshold for candidate selection
        selection_mode: Selection method ('confidence' for high-confidence samples, 'entropy' for high-uncertainty samples, 'basic_transition' for basic transition strategy, 'stratified_uncertainty' for stratified uncertainty sampling)
        basic_transition_f1_threshold: F1 threshold for basic transition (default: 0.2)
        basic_transition_confidence_threshold: Mean confidence threshold for basic transition (default: 0.9)
        basic_transition_variance_threshold: Std confidence threshold for basic transition (default: 0.12)
        auto_thresholds: Automatically calculate thresholds based on dataset characteristics (default: False)
        estimated_positive_pct: Estimated percentage of positive samples (0.0-1.0). Used with auto_thresholds.
        experiment_name: Optional experiment name to customize output directory
        seed: Random seed for reproducibility (default: None)
        with_ground_truth: Include ground truth evaluation columns (default: False)
        **dataset_kwargs: Additional dataset-specific configuration

    Returns:
        tuple: (predictions_file, candidates_file)
    """
    # Get unified config
    config = AudioLoopConfig(experiment_name=experiment_name, dataset=dataset_name)
    dataset_config = config.get_dataset_config()

    # Auto-detect dataset file if not provided
    if dataset_file is None:
        dataset_file = str(dataset_config.dataset_csv)

    # Validate class name and get class ID
    try:
        positive_class_id = dataset_config.name_to_id[positive_class_name]
    except KeyError:
        raise ValueError(
            f"Invalid class name: '{positive_class_name}'. Valid names: {list(dataset_config.name_to_id.keys())}"
        ) from None

    # Auto-generate negative class name if not provided
    if negative_class_name is None:
        negative_class_name = f"not_{positive_class_name}"

    # Create config with active learning parameters
    active_learning_config = AudioLoopConfig(
        experiment_name=experiment_name,
        dataset=dataset_name,
        total_candidates=total_candidates,
        positive_percentage=positive_percentage,
        min_confidence=min_confidence,
        selection_mode=selection_mode,
        basic_transition_f1_threshold=basic_transition_f1_threshold,
        basic_transition_confidence_threshold=basic_transition_confidence_threshold,
        basic_transition_variance_threshold=basic_transition_variance_threshold,
        auto_thresholds=auto_thresholds,
        estimated_positive_pct=estimated_positive_pct,
        with_ground_truth=with_ground_truth,
    )

    # Call the main function with clean signature
    return run_active_learning_cycle(
        config=active_learning_config,
        positive_class_id=positive_class_id,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        run_number=run_number,
        model_path=model_path,
        dataset_file=dataset_file,
        training_set_csv=training_set_csv,
        seed=seed,
        log_level=log_level,
    )


def main():
    """Main CLI entry point for active learning."""
    parser = argparse.ArgumentParser(
        description="Run active learning cycles for binary audio classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run drill detection cycle 1 (automatically uses outputs/model_v1.pt)
  python -m audioloop.active_learning --class-name Drill --run-number 1

  # Run speech detection cycle 2 (automatically uses outputs/model_v2.pt)
  python -m audioloop.active_learning --class-name Speech --run-number 2

  # Run with UrbanSound8K dataset
  python -m audioloop.active_learning --dataset urbansound8k --class-name siren --run-number 1

  # Run with class ID instead of name (UrbanSound8K only)
  python -m audioloop.active_learning --class-id 3 --run-number 1

  # Specify custom model path
  python -m audioloop.active_learning --class-name gun_shot --model custom_model.pt

  # Custom negative class name
  python -m audioloop.active_learning --class-name gun_shot --negative-name safe_sound --run-number 1

  # List all available classes
  python -m audioloop.active_learning --list-classes

  # Use custom experiment name (outputs go to outputs_myexp/)
  python -m audioloop.active_learning --class-name siren --run-number 1 --experiment myexp

  # Include ground truth evaluation columns (for datasets with known labels)
  python -m audioloop.active_learning --class-name Drill --run-number 1 --with-ground-truth
        """,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use ('fsd50k' or 'urbansound8k'). Can also be set via AUDIOLOOP_DATASET environment variable.",
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
        help="Path to trained model file (default: outputs/model_v{run_number}.pt or outputs_experiment/model_v{run_number}.pt)",
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
        help="Total number of candidates to select (default from config: 50)",
    )
    parser.add_argument(
        "--positive-pct",
        type=float,
        help="Percentage of candidates that should be positive predictions (default from config: 0.75)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold for candidate selection (default from config: 0.8)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="Path to dataset metadata CSV file (auto-detected from dataset if not specified)",
    )
    parser.add_argument(
        "--training-set",
        type=str,
        help="Path to training set CSV (default: training_sets/training_set_v{run_number}.csv or training_sets_experiment/training_set_v{run_number}.csv)",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["confidence", "entropy", "basic_transition", "stratified_uncertainty"],
        help="Selection method: 'confidence', 'entropy', 'basic_transition', or 'stratified_uncertainty' (default from config: confidence)",
    )

    # Basic transition configuration arguments
    parser.add_argument(
        "--basic-transition-f1-threshold",
        type=float,
        help="F1 threshold for basic transition (default from config: 0.2)",
    )
    parser.add_argument(
        "--basic-transition-confidence-threshold",
        type=float,
        help="Mean confidence threshold for basic transition (default from config: 0.9)",
    )
    parser.add_argument(
        "--basic-transition-variance-threshold",
        type=float,
        help="Std confidence threshold for basic transition (default from config: 0.12)",
    )
    parser.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="Automatically calculate BasicTransition thresholds based on dataset characteristics",
    )
    parser.add_argument(
        "--estimated-positive-pct",
        type=float,
        help="Estimated percentage of positive samples in dataset (0.0-1.0). Used with --auto-thresholds. If not specified, uses dataset defaults.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to customize output directory (default: outputs, with experiment: outputs_experiment)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility (default: None)",
    )
    parser.add_argument(
        "--with-ground-truth",
        action="store_true",
        help="Include ground truth evaluation columns (ground_truth, correct) in predictions CSV. Use for evaluation with labeled datasets.",
    )

    args = parser.parse_args()

    # Create config with CLI overrides (only when explicitly provided by user)
    try:
        config_overrides = {
            key: value
            for key, value in {
                "experiment_name": args.experiment,
                "dataset": args.dataset,
                "total_candidates": args.total_candidates,
                "positive_percentage": args.positive_pct,
                "min_confidence": args.min_confidence,
                "selection_mode": args.selection_mode,
                "basic_transition_f1_threshold": args.basic_transition_f1_threshold,
                "basic_transition_confidence_threshold": args.basic_transition_confidence_threshold,
                "basic_transition_variance_threshold": args.basic_transition_variance_threshold,
                "estimated_positive_pct": args.estimated_positive_pct,
            }.items()
            if value is not None
        }

        # Handle boolean flags separately (they're always provided by argparse)
        if args.auto_thresholds:
            config_overrides["auto_thresholds"] = True
        if args.with_ground_truth:
            config_overrides["with_ground_truth"] = True

        config = AudioLoopConfig(**config_overrides)
        dataset_config = config.get_dataset_config()
    except ValueError as e:
        parser.error(str(e))

    # Handle list classes
    if args.list_classes:
        dataset_config.list_classes()
        return

    # Default model path based on run_number if not specified
    if not args.model:
        args.model = str(config.get_model_path(args.run_number))
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
            positive_class_id = dataset_config.name_to_id[args.class_name]
        except KeyError:
            parser.error(
                f"Invalid class name: '{args.class_name}'. Valid names: {list(dataset_config.name_to_id.keys())}"
            )
    else:  # args.class_id is not None
        positive_class_id = args.class_id
        try:
            positive_class_name = dataset_config.vocabulary[positive_class_id]
        except KeyError:
            parser.error(
                f"Invalid class ID: {positive_class_id}. Valid IDs: {list(dataset_config.vocabulary.keys())}"
            )

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
    print(f"Ground truth evaluation: {'enabled' if config.with_ground_truth else 'disabled'}")

    # Run the active learning cycle with clean signature (reuse existing config)
    predictions_file, candidates_file = run_active_learning_cycle(
        config=config,
        positive_class_id=positive_class_id,
        positive_class_name=positive_class_name,
        negative_class_name=negative_class_name,
        run_number=args.run_number,
        model_path=args.model,
        dataset_file=args.dataset_file,
        training_set_csv=args.training_set,
        seed=args.seed,
    )

    print("\n✅ Active learning cycle completed!")
    print(f"📊 Predictions: {predictions_file}")
    print(f"🏷️  Candidates: {candidates_file}")
    print("\nNext steps:")
    print(f"1. Label candidates: python -m audioloop.label_audio {candidates_file}")
    print(
        f"2. Merge labels: python -m audioloop.merge_labels {config.get_training_set_path(args.run_number)} {candidates_file}"
    )


if __name__ == "__main__":
    main()
