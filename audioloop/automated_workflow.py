#!/usr/bin/env python3
"""
Automated Active Learning Workflow

This script automates the complete active learning cycle:
1. Train model from training set
2. Run active learning inference to generate candidates
3. Auto-label candidates (or pause for human labeling)
4. Merge labels back into expanded training set
5. Repeat for specified number of cycles

Usage:
    # Production workflow (manual labeling, no ground truth)
    python -m audioloop.automated_workflow --class-name siren --cycles 3

    # Evaluation workflow with manual labeling (ground truth available)
    python -m audioloop.automated_workflow --class-name dog_bark --cycles 2 --evaluation-mode

    # Evaluation workflow with auto-labeling (for testing/development)
    python -m audioloop.automated_workflow --class-name gun_shot --cycles 5 --evaluation-mode --auto-label
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

from audioloop.active_learning import run_active_learning_for_class
from audioloop.config import AudioLoopConfig
from audioloop.datasets.dataset_registry import list_available_datasets
from audioloop.merge_labels import merge_training_sets

# Import the APIs from the existing modules
from audioloop.training_core import run_training


def auto_label_candidates(
    candidates_file, dataset_name="fsd50k", audio_dir=None, log_level=logging.INFO
):
    """
    Automatically label candidates using ground truth data.

    Args:
        candidates_file: Path to candidates CSV file
        dataset_name: Dataset name (unused, kept for compatibility)
        audio_dir: Audio directory (unused, kept for compatibility)
        log_level: Logging level (logging.INFO for normal output, logging.WARNING for quiet)

    Returns:
        int: Number of samples that were labeled
    """
    try:
        from audioloop.utils.auto_labeling import auto_label_from_ground_truth

        results = auto_label_from_ground_truth(candidates_file, log_level=log_level)

        # Always show the summary (even in quiet mode)
        print(f"      Positive: {results['positive_count']}")
        print(f"      Negative: {results['negative_count']}")

        return results["total"]

    except Exception as e:
        print(f"   ❌ Error in auto-labeling: {e}")
        return 0


def prompt_for_manual_labeling(candidates_file):
    """
    Prompt user to manually label candidates and wait for completion.

    Args:
        candidates_file: Path to candidates CSV file

    Returns:
        int: Number of samples that were labeled (estimated)
    """
    print("👤 MANUAL LABELING REQUIRED:")
    print(f"   File: {candidates_file}")
    print(f"   Command: python -m audioloop.label_audio {candidates_file}")
    print()
    print("   Please run the labeling command in another terminal, then return here.")
    input("   Press Enter when labeling is complete...")

    # Count labeled samples
    labeled_count = 0
    try:
        with open(candidates_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("needs_human_label", "").strip():
                    labeled_count += 1
        print(f"   ✅ Found {labeled_count} manually labeled samples")
    except Exception as e:
        print(f"   ⚠️ Could not count labeled samples: {e}")
        labeled_count = 1  # Assume at least one sample was labeled

    return labeled_count


def run_automated_workflow(
    config,
    class_name,
    num_cycles=3,
    auto_label=True,
    evaluation_mode=False,
    audio_dir=None,
    seed=None,
):
    """
    Run the complete automated active learning workflow.

    Args:
        config: AudioLoopConfig instance with all configuration parameters
        class_name: Target class name (e.g., "siren", "dog_bark")
        num_cycles: Number of active learning cycles
        auto_label: Whether to auto-label (True) or prompt for manual labeling (False)
        evaluation_mode: Whether to enable evaluation mode with ground truth access (False)
        audio_dir: Custom audio directory (optional)
        seed: Random seed for reproducibility (default: None)

    Returns:
        list: Paths to training sets created during the workflow
    """
    print("\n" + "=" * 80)
    print("🚀 AUTOMATED ACTIVE LEARNING WORKFLOW")
    print("=" * 80)
    print(f"Target class: {class_name}")
    print(f"Dataset: {config.dataset}")
    print(f"Cycles: {num_cycles}")
    print(f"Evaluation mode: {evaluation_mode}")
    print(f"Auto-label: {auto_label}")
    print(f"Training epochs: {config.max_epochs}")
    print(f"Candidates per cycle: {config.total_candidates}")
    print(f"Selection strategy: {config.selection_mode}")
    print("=" * 80)

    # Check that initial training set exists
    initial_training_set = config.get_training_set_path(1)
    if not os.path.exists(initial_training_set):
        print(f"❌ Initial training set not found: {initial_training_set}")
        print("💡 Create it first with:")
        if config.experiment_name:
            print(
                f"   python -m audioloop.utils.create_bootstrap_set --class-name {class_name} --n 40 --experiment {config.experiment_name}"
            )
        else:
            print(
                f"   python -m audioloop.utils.create_bootstrap_set --class-name {class_name} --n 40"
            )
        return []

    training_sets = [initial_training_set]

    for cycle in range(1, num_cycles + 1):
        print("\n" + "█" * 80)
        print(f"🔄 CYCLE {cycle} of {num_cycles}")
        print("█" * 80)

        current_training_set = config.get_training_set_path(cycle)
        current_model = config.get_model_path(cycle)

        # Use the training set from the previous cycle if current doesn't exist
        if not os.path.exists(current_training_set) and len(training_sets) > 0:
            current_training_set = training_sets[-1]
            print(f"Using previous training set: {current_training_set}")

        # Step 1: Train model
        print("├─ Training model...")
        try:
            import logging

            final_accuracy = run_training(
                config=config,
                labels_file=current_training_set,
                version=cycle,
                model_path=current_model,
                log_level=logging.WARNING,  # Quiet mode
            )
            print(f"├─ Training model... ✓ ({final_accuracy * 100:.1f}% accuracy)")
        except Exception as e:
            print(f"├─ Training model... ❌ ({e})")
            break

        # Step 2: Run active learning
        print("├─ Selecting candidates...")
        try:
            import logging

            predictions_file, candidates_file = run_active_learning_for_class(
                positive_class_name=class_name,
                model_path=current_model,
                dataset_name=config.dataset,
                run_number=cycle,
                total_candidates=config.total_candidates,
                positive_percentage=config.positive_percentage,
                min_confidence=config.min_confidence,
                selection_mode=config.selection_mode,
                basic_transition_f1_threshold=config.basic_transition_f1_threshold,
                basic_transition_confidence_threshold=config.basic_transition_confidence_threshold,
                basic_transition_variance_threshold=config.basic_transition_variance_threshold,
                auto_thresholds=config.auto_thresholds,
                estimated_positive_pct=config.estimated_positive_pct,
                experiment_name=config.experiment_name,
                seed=seed,
                with_ground_truth=evaluation_mode,
                log_level=logging.WARNING,  # Quiet mode
            )
            # Count candidates for clean reporting
            try:
                import csv

                with open(candidates_file) as f:
                    candidate_count = sum(1 for _ in csv.DictReader(f))
                print(f"├─ Selecting candidates... ✓ ({candidate_count} samples)")
            except Exception:
                print("├─ Selecting candidates... ✓")
        except Exception as e:
            print(f"├─ Selecting candidates... ❌ ({e})")
            break

        # Step 3: Label candidates
        if evaluation_mode and auto_label:
            print("├─ Auto-labeling...")
            labeled_count = auto_label_candidates(
                candidates_file=candidates_file,
                dataset_name=config.dataset,
                audio_dir=audio_dir,
                log_level=logging.WARNING,
            )
            print("├─ Auto-labeling... ✓")
        else:
            print("\n🏷️  STEP 3: LABELING CANDIDATES")
            print("▼" * 50)
            labeled_count = prompt_for_manual_labeling(candidates_file)

        if labeled_count == 0:
            print("   ⚠️ No samples were labeled, ending workflow")
            break

        # Step 4: Merge labels into new training set
        print("├─ Merging labels...")
        try:
            new_training_set = merge_training_sets(
                original_csv=current_training_set,
                new_labels_csv=candidates_file,
                log_level=logging.WARNING,
            )
            training_sets.append(new_training_set)
            print(f"├─ Merging labels... ✓ (created {os.path.basename(new_training_set)})")
        except Exception as e:
            print(f"├─ Merging labels... ❌ ({e})")
            break

        # Show progress summary
        print("\n" + "▲" * 50)
        print(f"✅ CYCLE {cycle} COMPLETE")
        print(f"   📊 Added {labeled_count} new labeled samples")
        print(f"   📁 Next training set: {new_training_set}")
        print("▲" * 50)

        # Brief pause between cycles
        if cycle < num_cycles:
            time.sleep(1)

    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Completed {len(training_sets) - 1} cycles")
    print("Training sets created:")
    for i, ts in enumerate(training_sets):
        print(f"   v{i + 1}: {ts}")
    print("=" * 80)

    return training_sets


def main():
    parser = argparse.ArgumentParser(
        description="Automated active learning workflow for audio classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Production workflow (manual labeling, no ground truth)
  python -m audioloop.automated_workflow --class-name Drill --cycles 3

  # Evaluation workflow with manual labeling (ground truth available)
  python -m audioloop.automated_workflow --class-name Speech --cycles 2 --evaluation-mode

  # Evaluation workflow with auto-labeling (for testing/development)
  python -m audioloop.automated_workflow --class-name Music --cycles 3 --evaluation-mode --auto-label

  # Custom training parameters with evaluation mode
  python -m audioloop.automated_workflow --class-name Drill --cycles 2 --evaluation-mode --epochs 500 --batch-size 64

  # Custom stopping criteria (production)
  python -m audioloop.automated_workflow --class-name Explosion --cycles 2 --accuracy-floor 0.95 --patience 30

  # UrbanSound8K dataset with evaluation mode
  python -m audioloop.automated_workflow --class-name siren --cycles 2 --dataset urbansound8k --evaluation-mode --auto-label

  # Use basic transition strategy with evaluation
  python -m audioloop.automated_workflow --class-name Drill --cycles 3 --evaluation-mode --selection-mode basic_transition --auto-label

  # Use entropy-based selection (production)
  python -m audioloop.automated_workflow --class-name Speech --cycles 2 --selection-mode entropy

Prerequisites:
  1. Run: python -m audioloop.create_specs  (one-time setup)
  2. Run: python -m audioloop.utils.create_bootstrap_set --class-name <CLASS_NAME> --n 50
        """,
    )

    # Required arguments
    parser.add_argument(
        "--class-name", required=True, help="Target class name (e.g., siren, dog_bark)"
    )

    # Workflow parameters
    parser.add_argument("--cycles", type=int, help="Number of active learning cycles (default: 3)")
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Enable evaluation mode with ground truth access (for research/tuning)",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        help="Automatically label using ground truth (requires --evaluation-mode)",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        choices=None,  # Will be validated dynamically using registry
        help="Dataset name (default from config: fsd50k)",
    )
    parser.add_argument(
        "--audio-dir", help="Custom audio directory (uses dataset default if not specified)"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, help="Max training epochs per cycle (default from config: 1000)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Training batch size (default from config: 32)"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Learning rate (default from config: 0.001)"
    )

    # Stopping criteria parameters
    parser.add_argument(
        "--stopping-criterion",
        choices=["accuracy", "plateau", "hybrid"],
        help="Stopping criterion to use (default from config: hybrid)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="Patience for plateau stopping criterion (default from config: 25)",
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

    # Active learning parameters
    parser.add_argument(
        "--candidates",
        type=int,
        help="Number of candidates to select per cycle (default from config: 50)",
    )
    parser.add_argument(
        "--positive-pct",
        type=float,
        help="Target percentage of positive samples (default from config: 0.75)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold (default from config: 0.8)",
    )

    # Selection strategy parameters
    parser.add_argument(
        "--selection-mode",
        choices=["confidence", "entropy", "basic_transition", "stratified_uncertainty"],
        help="Selection strategy: 'confidence', 'entropy', 'basic_transition', or 'stratified_uncertainty' (default from config: confidence)",
    )
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
        help="Random seed for reproducibility (default from config: 42)",
    )
    parser.add_argument(
        "--class-weighting",
        type=str,
        help="Class weighting mode: 'adaptive' (inverse frequency), float 0.0-1.0 (fixed target positive ratio), or omit for no weighting",
    )

    args = parser.parse_args()

    # Validate flag combinations
    if args.auto_label and not args.evaluation_mode:
        parser.error("--auto-label requires --evaluation-mode")

    # Validate dataset choice using registry
    if args.dataset is not None:
        available_datasets = list_available_datasets()
        if args.dataset not in available_datasets:
            print(
                f"Error: Invalid dataset '{args.dataset}'. Available datasets: {', '.join(available_datasets)}"
            )
            sys.exit(1)

    # Create config with CLI overrides (config-first approach)
    config_overrides = {
        key: value
        for key, value in {
            "experiment_name": args.experiment,
            "dataset": args.dataset,
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "stopping_criterion_type": args.stopping_criterion,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "accuracy_floor": args.accuracy_floor,
            "total_candidates": args.candidates,
            "positive_percentage": args.positive_pct,
            "min_confidence": args.min_confidence,
            "selection_mode": args.selection_mode,
            "basic_transition_f1_threshold": args.basic_transition_f1_threshold,
            "basic_transition_confidence_threshold": args.basic_transition_confidence_threshold,
            "basic_transition_variance_threshold": args.basic_transition_variance_threshold,
            "estimated_positive_pct": args.estimated_positive_pct,
            "seed": args.seed,
        }.items()
        if value is not None
    }

    # Handle boolean flags separately (they're always provided by argparse)
    if args.auto_thresholds:
        config_overrides["auto_thresholds"] = True
    if args.evaluation_mode:
        config_overrides["with_ground_truth"] = True

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

    # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    # Save command line arguments to the output directory
    command_info = {
        "command": " ".join(sys.argv),
        "arguments": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save JSON file with full information
    command_file = os.path.join(config.output_dir, "command_line.json")
    with open(command_file, "w") as f:
        json.dump(command_info, f, indent=2)

    # Save plain text file with just the command
    command_txt_file = os.path.join(config.output_dir, "command_line.txt")
    with open(command_txt_file, "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    print(f"📝 Command line saved to: {command_file}")
    print(f"📝 Command line text saved to: {command_txt_file}")

    # Validate arguments (use config values for validation since CLI may be None)
    if config.positive_percentage < 0 or config.positive_percentage > 1:
        print("❌ --positive-pct must be between 0 and 1")
        return 1

    if config.min_confidence < 0 or config.min_confidence > 1:
        print("❌ --min-confidence must be between 0 and 1")
        return 1

    # Check prerequisites
    if not os.path.exists("data/all_specs"):
        print("❌ Spectrograms not found!")
        print("💡 Run this first: python -m audioloop.create_specs")
        return 1

    # Run the workflow
    try:
        training_sets = run_automated_workflow(
            config=config,
            class_name=args.class_name,
            num_cycles=args.cycles if args.cycles is not None else 3,
            auto_label=args.auto_label,
            evaluation_mode=args.evaluation_mode,
            audio_dir=args.audio_dir,
            seed=args.seed,
        )

        if training_sets:
            print("\n📈 WORKFLOW SUMMARY:")
            print(f"   Initial training set: {training_sets[0]}")
            print(f"   Final training set: {training_sets[-1]}")
            print(f"   Total cycles completed: {len(training_sets) - 1}")

            # Suggest next steps
            print("\n💡 NEXT STEPS:")
            print("   • Run metrics: python -m audioloop.track_metrics --plot")
            print("   • Clean outputs: python -m audioloop.clean_outputs")
            return 0
        print("❌ Workflow failed to complete any cycles")
        return 1

    except KeyboardInterrupt:
        print("\n⏹️ Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
