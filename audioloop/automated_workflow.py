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
import shutil
import sys
import time
from pathlib import Path

from audioloop.active_learning import run_active_learning_for_class
from audioloop.config import AudioLoopConfig
from audioloop.datasets.dataset_registry import list_available_datasets
from audioloop.merge_labels import merge_training_sets
from audioloop.utils.candidate_metrics import load_candidate_metrics_history
from audioloop.utils.cycle_stopping_criteria import create_cycle_stopping_criterion

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
    start_cycle=1,
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
        num_cycles: Number of active learning cycles to run through
        start_cycle: Cycle number to start from (default: 1, for resuming workflows)
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
    if config.subset_csv:
        print(f"Subset: {config.subset_csv.name}")
    if start_cycle == 1:
        print(f"Cycles: 1 to {num_cycles}")
    else:
        print(f"Cycles: {start_cycle} to {num_cycles} (resuming)")
    print(f"Evaluation mode: {evaluation_mode}")
    print(f"Auto-label: {auto_label}")
    print(f"Training epochs: {config.max_epochs}")
    print(f"Candidates per cycle: {config.total_candidates}")
    print(f"Selection strategy: {config.selection_mode}")

    # Format class weighting for display
    if config.class_weighting is None:
        class_weighting_str = "none"
    elif config.class_weighting == "adaptive":
        class_weighting_str = "adaptive"
    else:
        class_weighting_str = f"fixed ({config.class_weighting:.2f} target positive ratio)"
    print(f"Class weighting: {class_weighting_str}")

    print("=" * 80)

    # Verify the starting training set exists
    initial_training_set = config.get_training_set_path(start_cycle)
    if not os.path.exists(initial_training_set):
        if start_cycle == 1:
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
        else:
            # Find the last existing training set to suggest the right cycle
            last_existing_training_set = 0
            for i in range(1, start_cycle):
                if os.path.exists(config.get_training_set_path(i)):
                    last_existing_training_set = i
                else:
                    break

            print(f"❌ Cannot start at cycle {start_cycle}: {initial_training_set} not found")
            if last_existing_training_set > 0:
                print(
                    f"💡 Last available training set is v{last_existing_training_set}. Retry with --start-cycle {last_existing_training_set}"
                )
            else:
                print(
                    "💡 No training sets found. Start with cycle 1 or create initial training set."
                )
        return []

    if start_cycle > 1:
        print(f"📂 Resuming from cycle {start_cycle}")
        print()

    # Track training sets created during this workflow run
    training_sets = [initial_training_set]

    # Initialize stopping criterion (will create after first cycle when metrics exist)
    stopping_criterion = None

    for cycle in range(start_cycle, num_cycles + 1):
        print("\n" + "█" * 80)
        print(f"🔄 CYCLE {cycle} of {num_cycles}")
        print("█" * 80)

        current_training_set = config.get_training_set_path(cycle)
        current_model = config.get_model_path(cycle)

        # Step 1: Train model
        print("├─ Training model...")
        try:
            import logging

            final_accuracy, num_epochs = run_training(
                config=config,
                labels_file=current_training_set,
                version=cycle,
                model_path=current_model,
                log_level=logging.WARNING,  # Quiet mode
            )
            print(
                f"├─ Training model... ✓ ({final_accuracy * 100:.1f}% accuracy, {num_epochs} epochs)"
            )
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
                subset_csv=str(config.subset_csv) if config.subset_csv else None,
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
                config=config,
                log_level=logging.WARNING,
            )
            training_sets.append(new_training_set)
            print(f"├─ Merging labels... ✓ (created {os.path.basename(new_training_set)})")
        except Exception as e:
            print(f"├─ Merging labels... ❌ ({e})")
            break

        # Check stopping criterion
        if config.cycle_stopping_strategy != "none":
            metrics_history = load_candidate_metrics_history(config.output_dir)

            # Create or update criterion
            if stopping_criterion is None and metrics_history:
                stopping_criterion = create_cycle_stopping_criterion(config, metrics_history)
            elif stopping_criterion:
                stopping_criterion.metrics_history = metrics_history

            # Check if should stop
            if stopping_criterion and stopping_criterion.should_stop(cycle):
                best_cycle = stopping_criterion.get_best_cycle()

                # Copy best model
                best_model_src = config.get_model_path(best_cycle)
                best_model_dst = config.output_dir / "model_best.pt"
                shutil.copy(best_model_src, best_model_dst)

                print(f"\n🛑 STOPPING CRITERION MET")
                print(f"   Strategy: {config.cycle_stopping_strategy}")
                print(f"   Best cycle: {best_cycle}")
                print(f"   Copied model_v{best_cycle}.pt -> model_best.pt")
                break

        # Show progress summary
        print("\n" + "▲" * 50)
        print(f"✅ CYCLE {cycle} COMPLETE")
        print(f"   📊 Added {labeled_count} new labeled samples")
        print(f"   📁 Next training set: {new_training_set}")

        # Helper function to display confusion matrix
        def display_confusion_matrix(metrics):
            """Display confusion matrix for candidate metrics with recall/precision"""
            tp = int(metrics.get('true_positives', 0))
            fp = int(metrics.get('false_positives', 0))
            tn = int(metrics.get('true_negatives', 0))
            fn = int(metrics.get('false_negatives', 0))
            total = tp + fp + tn + fn

            if total == 0:
                return

            print(f"      Confusion Matrix ({total} labeled candidates):")
            print(f"                       Predicted")
            print(f"                    Positive  Negative")
            print(f"         Actual Pos    {tp:>4}      {fn:>4}     (recall: {metrics['recall']:.3f})")
            print(f"                Neg    {fp:>4}      {tn:>4}")
            print(f"                     -------  -------")
            print(f"         Precision   {metrics['precision']:.3f}")

        # Display detailed stopping status
        if stopping_criterion and metrics_history and cycle in metrics_history:
            current_metrics = metrics_history[cycle]

            if config.cycle_stopping_strategy == "label":
                # LabelMode: Show F1 tracking
                rolling_f1 = stopping_criterion._calculate_rolling_avg(
                    "f1_score", config.cycle_window, cycle
                )
                rolling_std = stopping_criterion._calculate_rolling_std(
                    "f1_score", config.cycle_window, cycle
                )

                print(f"   📈 Stopping (label mode):")
                print(f"      F1 (current): {current_metrics['f1_score']:.3f}")
                print(
                    f"      F1 (rolling avg): {rolling_f1:.3f} (window={config.cycle_window})"
                )
                print(
                    f"      F1 (stability): std={rolling_std:.3f} (threshold={config.cycle_std_threshold})"
                )
                print(
                    f"      Patience: {stopping_criterion.cycles_without_improvement}/{config.cycle_patience}"
                )
                print(
                    f"      Best cycle: {stopping_criterion.get_best_cycle()} "
                    f"(rolling avg={stopping_criterion.best_rolling_avg:.3f})"
                )
                print()
                display_confusion_matrix(current_metrics)

            elif config.cycle_stopping_strategy == "search":
                # SearchMode: Show recall + precision tracking
                rolling_recall = stopping_criterion._calculate_rolling_avg(
                    "recall", config.cycle_window, cycle
                )
                rolling_precision = stopping_criterion._calculate_rolling_avg(
                    "precision", config.cycle_window, cycle
                )
                rolling_std = stopping_criterion._calculate_rolling_std(
                    "recall", config.cycle_window, cycle
                )

                print(f"   📈 Stopping (search mode):")
                print(f"      Recall (current): {current_metrics['recall']:.3f}")
                print(
                    f"      Recall (rolling avg): {rolling_recall:.3f} (window={config.cycle_window})"
                )
                print(
                    f"      Precision (rolling avg): {rolling_precision:.3f} "
                    f"(floor={stopping_criterion._precision_floor:.3f})"
                )
                print(f"      Recall (stability): std={rolling_std:.3f} (threshold=0.10)")
                print(
                    f"      Patience: {stopping_criterion.cycles_without_improvement}/{config.cycle_patience}"
                )
                print(
                    f"      Best cycle: {stopping_criterion.get_best_cycle()} "
                    f"(rolling recall={stopping_criterion.best_rolling_avg:.3f})"
                )
                print()
                display_confusion_matrix(current_metrics)

        # Display confusion matrix even when stopping criteria are disabled
        elif config.cycle_stopping_strategy == "none":
            metrics_history = load_candidate_metrics_history(config.output_dir)
            if metrics_history and cycle in metrics_history:
                print(f"   📊 Candidate Performance:")
                display_confusion_matrix(metrics_history[cycle])

        print("▲" * 50)

        # Brief pause between cycles
        if cycle < num_cycles:
            time.sleep(1)

    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Completed {len(training_sets) - 1} cycles")
    print("Training sets from this run:")
    for i, ts in enumerate(training_sets, start=start_cycle):
        print(f"   v{i}: {ts}")
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

  # Use AudioSet subset for efficient workflow
  python -m audioloop.automated_workflow --dataset audioset --class-name Dog \\
      --subset-csv subsets/audioset_dog_10k.csv --cycles 5 --evaluation-mode --auto-label

  # Use basic transition strategy with evaluation
  python -m audioloop.automated_workflow --class-name Drill --cycles 3 --evaluation-mode --selection-mode basic_transition --auto-label

  # Use entropy-based selection (production)
  python -m audioloop.automated_workflow --class-name Speech --cycles 2 --selection-mode entropy

  # Resume interrupted workflow from cycle 16 (run cycles 16-30)
  python -m audioloop.automated_workflow --class-name Drill --cycles 30 --start-cycle 16 --experiment myexp --evaluation-mode --auto-label

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
    parser.add_argument(
        "--cycles", type=int, help="Number of active learning cycles to run through (default: 3)"
    )
    parser.add_argument(
        "--start-cycle",
        type=int,
        default=1,
        help="Cycle number to start from (default: 1). Use this to resume interrupted workflows.",
    )
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
        "--subset-csv",
        type=str,
        help="Path to subset CSV to restrict active learning to specific files (e.g., from create_subset)",
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
    parser.add_argument(
        "--no-lr-scheduler",
        action="store_true",
        help="Disable learning rate scheduler (enabled by default)"
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        help="LR reduction factor for scheduler (default: 0.5)"
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        help="Scheduler patience in epochs (default: 5)"
    )
    parser.add_argument(
        "--model-type", type=str, help="Model architecture to use (default from config: cnn5layer)"
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

    # Cycle stopping criteria (cross-cycle stopping based on candidate metrics)
    parser.add_argument(
        "--cycle-stopping-strategy",
        choices=["none", "label", "search"],
        help="Cycle stopping strategy: 'none' (default, no stopping), 'label' (optimize F1), or 'search' (optimize recall with precision floor)",
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
        help="Stratify candidate selection by predicted class (0.0-1.0). Default: None (pure entropy, no stratification)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold (default from config: 0.8)",
    )

    # Selection strategy parameters
    parser.add_argument(
        "--selection-mode",
        choices=["confidence", "entropy", "mixed_entropy", "basic_transition", "stratified_uncertainty", "random"],
        help="Selection strategy (default: mixed_entropy)",
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
        help="Experiment name to customize output directory (default: outputs, with experiment: outputs/experiment)",
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
            "subset_csv": Path(args.subset_csv) if args.subset_csv else None,
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "use_lr_scheduler": not args.no_lr_scheduler,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "model_type": args.model_type,
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
            "cycle_stopping_strategy": args.cycle_stopping_strategy,
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
    if config.positive_percentage is not None and (config.positive_percentage < 0 or config.positive_percentage > 1):
        print("❌ --positive-pct must be between 0 and 1 (or omit for no stratification)")
        return 1

    if config.min_confidence < 0 or config.min_confidence > 1:
        print("❌ --min-confidence must be between 0 and 1")
        return 1

    # Check prerequisites
    if not os.path.exists("data/all_specs"):
        print("❌ Spectrograms not found!")
        print("💡 Run this first: python -m audioloop.create_specs")
        return 1

    # Validate start_cycle
    if args.start_cycle < 1:
        print("❌ --start-cycle must be >= 1")
        return 1

    num_cycles = args.cycles if args.cycles is not None else 3
    if args.start_cycle > num_cycles:
        print(f"❌ --start-cycle ({args.start_cycle}) must be <= --cycles ({num_cycles})")
        return 1

    # Run the workflow
    try:
        training_sets = run_automated_workflow(
            config=config,
            class_name=args.class_name,
            num_cycles=num_cycles,
            start_cycle=args.start_cycle,
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
            # Include --experiment flag in suggestions if using an experiment
            experiment_flag = (
                f" --experiment {config.experiment_name}" if config.experiment_name else ""
            )
            print(f"   • Run metrics: python -m audioloop.track_metrics --plot{experiment_flag}")
            print(f"   • Clean outputs: python -m audioloop.clean_outputs{experiment_flag}")
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
