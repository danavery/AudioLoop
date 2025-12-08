import argparse
import logging
import re

from .config import AudioLoopConfig
from .models.model_registry import get_model_descriptions, list_available_models
from .training_core import run_training, set_seed, train_epoch

# Re-export for backward compatibility
__all__ = ["main", "run_training", "set_seed", "train_epoch"]


def list_models():
    """Print available models and exit."""
    print("Available model types:")
    for model_type, description in get_model_descriptions().items():
        print(f"  {model_type}: {description}")


def main():
    """Main CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a binary audio classification model")
    parser.add_argument("labels_file", nargs="?", help="Path to CSV file with training labels")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save trained model (default: auto-generated in outputs/ or outputs/experiment/)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        help="Model version number (default: auto-detected from training set filename)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, help="Maximum training epochs (default from config: 1000)"
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed (default from config: 42)")
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size (default from config: 32)")
    parser.add_argument(
        "--model-type",
        choices=list_available_models(),
        help="Model type to use (default from config: cnn5layer). Use --list-models to see all options",
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available model types and exit"
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, help="Learning rate (default from config: 0.001)"
    )
    parser.add_argument(
        "--stopping-criterion",
        type=str,
        choices=["accuracy", "plateau"],
        help="Stopping criterion to use (default from config: plateau)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Patience for plateau stopping criterion (default from config: 20)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        help="Minimum delta for plateau stopping criterion (default from config: 0.01)",
    )

    parser.add_argument(
        "--accuracy-floor",
        type=float,
        help="Only count plateau patience when accuracy >= this threshold (default from config: None)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use ('fsd50k', 'urbansound8k', 'audioset'). Can also be set via AUDIOLOOP_DATASET environment variable.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to customize output directory (default: outputs, with experiment: outputs/experiment)",
    )
    parser.add_argument(
        "--class-weighting",
        type=str,
        help="Class weighting mode: 'adaptive' (inverse frequency), float 0.0-1.0 (fixed target positive ratio), or omit for no weighting",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress training progress output",
    )

    args = parser.parse_args()

    # Handle list-models command
    if args.list_models:
        list_models()
        return

    # Check if labels_file is provided when not listing models
    if args.labels_file is None:
        parser.error("labels_file is required unless using --list-models")

    # Auto-detect version from training set filename if not specified
    if args.version is None:
        # Try to extract version from filename like training_set_v2.csv
        match = re.search(r"_v(\d+)\.csv$", args.labels_file)
        if match:
            args.version = int(match.group(1))
            print(f"Auto-detected version {args.version} from training set filename")
        else:
            args.version = 1
            print("Could not detect version from filename, defaulting to version 1")

    # Create config with CLI overrides (only when explicitly provided by user)
    config_overrides = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "experiment_name": args.experiment,
            "max_epochs": args.epochs,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_type": args.model_type,
            "stopping_criterion_type": args.stopping_criterion,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "accuracy_floor": args.accuracy_floor,
        }.items()
        if value is not None
    }

    # Handle class_weighting (can be "adaptive", a float, or None)
    if args.class_weighting is not None:
        if args.class_weighting == "adaptive":
            config_overrides["class_weighting"] = "adaptive"
        else:
            # Try to parse as float
            try:
                config_overrides["class_weighting"] = float(args.class_weighting)
            except ValueError:
                parser.error(
                    f"--class-weighting must be 'adaptive' or a float between 0.0 and 1.0, got: {args.class_weighting}"
                )

    config = AudioLoopConfig(**config_overrides)

    # Set log level based on quiet flag
    log_level = logging.WARNING if args.quiet else logging.INFO

    # Run training with clean signature
    accuracy, num_epochs = run_training(
        config=config,
        labels_file=args.labels_file,
        version=args.version,
        model_path=args.output,
        log_level=log_level,
    )
    print(f"\nFinal training accuracy: {accuracy:.4f} ({num_epochs} epochs)")


if __name__ == "__main__":
    main()
