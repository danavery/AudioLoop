import argparse
import re

from .training_core import run_training, set_seed, train_epoch
from .utils.stopping_criteria import AccuracyCriterion, PlateauCriterion

# Re-export for backward compatibility
__all__ = ["main", "run_training", "set_seed", "train_epoch"]


def main():
    """Main CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train a binary audio classification model")
    parser.add_argument("labels_file", help="Path to CSV file with training labels")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to save trained model (default: auto-generated in outputs/ or outputs_experiment/)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        help="Model version number (default: auto-detected from training set filename)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1000, help="Maximum training epochs (default: 1000)"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--specs-dir",
        default="data/all_specs",
        help="Directory containing spectrogram files (default: data/all_specs)",
    )
    parser.add_argument(
        "--no-batchnorm",
        action="store_true",
        help="Disable BatchNorm (auto-disabled for <100 samples)",
    )
    parser.add_argument(
        "--stopping-criterion",
        type=str,
        choices=["accuracy", "plateau"],
        default="plateau",
        help="Stopping criterion to use (default: plateau)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Patience for plateau stopping criterion (default: 20)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.01,
        help="Minimum delta for plateau stopping criterion (default: 0.01)",
    )

    parser.add_argument(
        "--accuracy-floor",
        type=float,
        default=None,
        help="Only count plateau patience when accuracy >= this threshold (default: None)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name to customize output directory (default: outputs, with experiment: outputs_experiment)",
    )

    args = parser.parse_args()

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

    # Create stopping criterion based on CLI arguments
    if args.stopping_criterion == "accuracy":
        stopping_criterion = AccuracyCriterion(max_epochs=args.epochs)
    elif args.stopping_criterion == "plateau":
        stopping_criterion = PlateauCriterion(
            patience=args.patience,
            min_delta=args.min_delta,
            max_epochs=args.epochs,
            accuracy_floor=args.accuracy_floor,
        )
    else:
        stopping_criterion = PlateauCriterion(
            patience=args.patience,
            min_delta=args.min_delta,
            max_epochs=args.epochs,
            accuracy_floor=args.accuracy_floor,
        )

    # Use experiment-aware model path if output not specified
    if args.output is None:
        output_dir = f"outputs_{args.experiment}" if args.experiment else "outputs"
        args.output = f"{output_dir}/model_v{args.version}.pt"

    # Run training with CLI arguments
    accuracy = run_training(
        labels_file=args.labels_file,
        max_epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        specs_dir=args.specs_dir,
        model_path=args.output,
        version=args.version,
        use_batchnorm=False if args.no_batchnorm else None,
        stopping_criterion=stopping_criterion,
    )
    print(f"\nFinal training accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
