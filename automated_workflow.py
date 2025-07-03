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
    # Automated workflow with auto-labeling (for testing/development)
    python automated_workflow.py --class-name siren --cycles 3 --auto-label

    # Semi-automated workflow (pause for human labeling)
    python automated_workflow.py --class-name dog_bark --cycles 2

    # Custom parameters
    python automated_workflow.py --class-name gun_shot --cycles 5 --epochs 500 --candidates 100 --auto-label
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Import the APIs from the existing modules
from audioloop.simple_train import run_training
from audioloop.active_learning import run_active_learning_for_class
from audioloop.merge_labels import merge_training_sets
from audioloop.label_audio import SimpleAudioLabeler


def auto_label_candidates(candidates_file, dataset_name="urbansound8k", audio_dir=None):
    """
    Automatically label candidates using the auto-label functionality.

    Args:
        candidates_file: Path to candidates CSV file
        dataset_name: Dataset name for proper auto-labeling
        audio_dir: Audio directory (optional, uses default if None)

    Returns:
        int: Number of samples that were labeled
    """
    print(f"📝 Auto-labeling candidates: {candidates_file}")

    try:
        # Create the labeler instance
        labeler = SimpleAudioLabeler(
            candidates_csv=candidates_file,
            dataset_name=dataset_name,
            audio_dir=audio_dir
        )

        # Run auto-labeling
        labeler.auto_label()

        # Count labeled samples by reading the file back
        labeled_count = 0
        with open(candidates_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('needs_human_label', '').strip():
                    labeled_count += 1

        print(f"   ✅ Auto-labeled {labeled_count} samples")
        return labeled_count

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
    print(f"👤 MANUAL LABELING REQUIRED:")
    print(f"   File: {candidates_file}")
    print(f"   Command: python -m audioloop.label_audio {candidates_file}")
    print()
    print("   Please run the labeling command in another terminal, then return here.")
    input("   Press Enter when labeling is complete...")

    # Count labeled samples
    labeled_count = 0
    try:
        with open(candidates_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('needs_human_label', '').strip():
                    labeled_count += 1
        print(f"   ✅ Found {labeled_count} manually labeled samples")
    except Exception as e:
        print(f"   ⚠️ Could not count labeled samples: {e}")
        labeled_count = 1  # Assume at least one sample was labeled

    return labeled_count


def run_automated_workflow(
    class_name,
    num_cycles=3,
    auto_label=True,
    dataset_name="urbansound8k",
    audio_dir=None,
    # Training parameters
    epochs=1000,
    batch_size=32,
    learning_rate=1e-3,
    # Active learning parameters
    total_candidates=50,
    positive_percentage=0.75,
    min_confidence=0.8,
):
    """
    Run the complete automated active learning workflow.

    Args:
        class_name: Target class name (e.g., "siren", "dog_bark")
        num_cycles: Number of active learning cycles
        auto_label: Whether to auto-label (True) or prompt for manual labeling (False)
        dataset_name: Dataset name ("urbansound8k" or "fsd50k")
        audio_dir: Custom audio directory (optional)
        epochs: Max training epochs per cycle
        batch_size: Training batch size
        learning_rate: Training learning rate
        total_candidates: Number of candidates to select per cycle
        positive_percentage: Target percentage of positive samples in candidates
        min_confidence: Minimum confidence threshold for candidate selection

    Returns:
        list: Paths to training sets created during the workflow
    """
    print("🚀 AUTOMATED ACTIVE LEARNING WORKFLOW")
    print("=" * 60)
    print(f"Target class: {class_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Cycles: {num_cycles}")
    print(f"Auto-label: {auto_label}")
    print(f"Training epochs: {epochs}")
    print(f"Candidates per cycle: {total_candidates}")
    print("=" * 60)

    # Check that initial training set exists
    initial_training_set = "training_sets/training_set_v1.csv"
    if not os.path.exists(initial_training_set):
        print(f"❌ Initial training set not found: {initial_training_set}")
        print("💡 Create it first with:")
        print(f"   python -m audioloop.utils.start_labeling --class-name {class_name} --n 40")
        return []

    training_sets = [initial_training_set]

    for cycle in range(1, num_cycles + 1):
        print(f"\n🔄 CYCLE {cycle}")
        print("-" * 40)

        current_training_set = f"training_sets/training_set_v{cycle}.csv"
        current_model = f"outputs/model_v{cycle}.pt"

        # Use the training set from the previous cycle if current doesn't exist
        if not os.path.exists(current_training_set) and len(training_sets) > 0:
            current_training_set = training_sets[-1]
            print(f"Using previous training set: {current_training_set}")

        # Step 1: Train model
        print(f"Step 1: Training model v{cycle}...")
        print(f"   Training set: {current_training_set}")
        print(f"   Model output: {current_model}")

        try:
            final_accuracy = run_training(
                labels_file=current_training_set,
                max_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                model_path=current_model,
                version=cycle,
            )
            print(f"   ✅ Model trained (final accuracy: {final_accuracy:.3f})")
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            break

        # Step 2: Run active learning
        print(f"Step 2: Running active learning cycle {cycle}...")
        try:
            predictions_file, candidates_file = run_active_learning_for_class(
                positive_class_name=class_name,
                model_path=current_model,
                dataset_name=dataset_name,
                run_number=cycle,
                total_candidates=total_candidates,
                positive_percentage=positive_percentage,
                min_confidence=min_confidence,
            )
            print(f"   ✅ Generated candidates: {candidates_file}")
        except Exception as e:
            print(f"   ❌ Active learning failed: {e}")
            break

        # Step 3: Label candidates
        print(f"Step 3: Labeling candidates...")
        if auto_label:
            labeled_count = auto_label_candidates(
                candidates_file=candidates_file,
                dataset_name=dataset_name,
                audio_dir=audio_dir
            )
        else:
            labeled_count = prompt_for_manual_labeling(candidates_file)

        if labeled_count == 0:
            print("   ⚠️ No samples were labeled, ending workflow")
            break

        # Step 4: Merge labels into new training set
        print(f"Step 4: Merging labels...")
        try:
            new_training_set = merge_training_sets(
                original_csv=current_training_set,
                new_labels_csv=candidates_file
            )
            training_sets.append(new_training_set)
            print(f"   ✅ Created: {new_training_set}")
        except Exception as e:
            print(f"   ❌ Label merging failed: {e}")
            break

        # Show progress summary
        print(f"Cycle {cycle} complete:")
        print(f"   📊 Added {labeled_count} new labeled samples")
        print(f"   📁 Next training set: {new_training_set}")

        # Brief pause between cycles
        if cycle < num_cycles:
            time.sleep(1)

    print(f"\n🎉 WORKFLOW COMPLETE!")
    print(f"Completed {len(training_sets) - 1} cycles")
    print(f"Training sets created:")
    for i, ts in enumerate(training_sets):
        print(f"   v{i + 1}: {ts}")

    return training_sets


def main():
    parser = argparse.ArgumentParser(
        description="Automated active learning workflow for audio classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully automated workflow (auto-labeling for testing)
  python automated_workflow.py --class-name siren --cycles 3 --auto-label

  # Semi-automated (pause for manual labeling)
  python automated_workflow.py --class-name dog_bark --cycles 2

  # Custom training parameters
  python automated_workflow.py --class-name gun_shot --cycles 3 --epochs 500 --batch-size 64

  # Custom active learning parameters
  python automated_workflow.py --class-name jackhammer --cycles 2 --candidates 100 --positive-pct 0.8

  # FSD50K dataset
  python automated_workflow.py --class-name Drill --cycles 2 --dataset fsd50k --auto-label

Prerequisites:
  1. Run: python -m audioloop.create_all_specs  (one-time setup)
  2. Run: python -m audioloop.utils.start_labeling --class-name <CLASS_NAME> --n 40
        """
    )

    # Required arguments
    parser.add_argument('--class-name', required=True,
                       help='Target class name (e.g., siren, dog_bark)')

    # Workflow parameters
    parser.add_argument('--cycles', type=int, default=3,
                       help='Number of active learning cycles (default: 3)')
    parser.add_argument('--auto-label', action='store_true',
                       help='Automatically label using ground truth (for testing)')

    # Dataset parameters
    parser.add_argument('--dataset', default='urbansound8k',
                       choices=['urbansound8k', 'fsd50k'],
                       help='Dataset name (default: urbansound8k)')
    parser.add_argument('--audio-dir',
                       help='Custom audio directory (uses dataset default if not specified)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Max training epochs per cycle (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')

    # Active learning parameters
    parser.add_argument('--candidates', type=int, default=50,
                       help='Number of candidates to select per cycle (default: 50)')
    parser.add_argument('--positive-pct', type=float, default=0.75,
                       help='Target percentage of positive samples (default: 0.75)')
    parser.add_argument('--min-confidence', type=float, default=0.8,
                       help='Minimum confidence threshold (default: 0.8)')

    args = parser.parse_args()

    # Validate arguments
    if args.positive_pct < 0 or args.positive_pct > 1:
        print("❌ --positive-pct must be between 0 and 1")
        return 1

    if args.min_confidence < 0 or args.min_confidence > 1:
        print("❌ --min-confidence must be between 0 and 1")
        return 1

    # Check prerequisites
    if not os.path.exists("data/all_specs"):
        print("❌ Spectrograms not found!")
        print("💡 Run this first: python -m audioloop.create_all_specs")
        return 1

    # Run the workflow
    try:
        training_sets = run_automated_workflow(
            class_name=args.class_name,
            num_cycles=args.cycles,
            auto_label=args.auto_label,
            dataset_name=args.dataset,
            audio_dir=args.audio_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            total_candidates=args.candidates,
            positive_percentage=args.positive_pct,
            min_confidence=args.min_confidence,
        )

        if training_sets:
            print(f"\n📈 WORKFLOW SUMMARY:")
            print(f"   Initial training set: {training_sets[0]}")
            print(f"   Final training set: {training_sets[-1]}")
            print(f"   Total cycles completed: {len(training_sets) - 1}")

            # Suggest next steps
            print(f"\n💡 NEXT STEPS:")
            print(f"   • Run metrics: python -m audioloop.track_metrics --plot")
            print(f"   • Clean outputs: python -m audioloop.clean_outputs")
            return 0
        else:
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
