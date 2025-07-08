#!/usr/bin/env python3
"""
Complete Active Learning Workflow Example

This script demonstrates the full active learning cycle including:
1. Running active learning to generate candidates
2. Simulating human labeling
3. Merging labels back into training set
4. Running subsequent cycles

NOTE: For convenience, consider the automated workflow instead:
    python automated_workflow.py --class-name siren --cycles 2 --auto-label

Usage:
    python example_workflow.py --class-name siren --cycles 2
"""

import argparse
import csv
import os
import random

from audioloop.active_learning import run_active_learning_for_class
from audioloop.merge_labels import merge_training_sets


def simulate_human_labeling(candidates_file, error_rate=0.1):
    """
    Simulate human labeling by adding labels to the needs_human_label column.

    In real usage, a human would manually fill in this column by listening
    to audio samples and deciding if they match the target class.

    Args:
        candidates_file: Path to candidates CSV file
        error_rate: Simulation parameter - rate of "human errors"

    Returns:
        int: Number of samples labeled
    """
    print(f"📝 Simulating human labeling for: {candidates_file}")

    # Read candidates file
    rows = []
    fieldnames = None
    with open(candidates_file) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            # Simulate human decision-making
            # In real life, human would listen to audio and decide
            predicted_label = int(row["predicted_label"])
            confidence = float(row["confidence"])

            # Simulate human labeling with occasional errors
            # Human disagrees with model at error_rate, otherwise agrees
            human_label = 1 - predicted_label if random.random() < error_rate else predicted_label

            # Only label high-confidence predictions (simulate human effort)
            if confidence >= 0.7:
                row["needs_human_label"] = str(human_label)
            else:
                row["needs_human_label"] = ""  # Human skips uncertain cases

            rows.append(row)

    # Write back with human labels
    with open(candidates_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labeled_count = sum(1 for row in rows if row["needs_human_label"])
    print(f"   → Labeled {labeled_count} samples")
    return labeled_count


def run_complete_workflow(
    class_name, initial_model, num_cycles=2, simulate_human=True, selection_mode="confidence"
):
    """
    Run complete active learning workflow with multiple cycles.

    Args:
        class_name: Target class name (e.g., "siren", "dog_bark")
        initial_model: Path to initial trained model
        num_cycles: Number of active learning cycles to run
        simulate_human: Whether to simulate human labeling (for demo)
        selection_mode: Selection strategy ('confidence', 'entropy', 'basic_transition')

    Returns:
        list: Training set paths for each cycle
    """
    print(f"🚀 Starting Active Learning Workflow: {class_name}")
    print(f"   Initial model: {initial_model}")
    print(f"   Cycles: {num_cycles}")
    print(f"   Selection strategy: {selection_mode}")
    print("=" * 60)

    training_sets = []
    current_model = initial_model

    for cycle in range(1, num_cycles + 1):
        print(f"\n🔄 CYCLE {cycle}")
        print("-" * 30)

        # Step 1: Run active learning cycle
        print("Step 1: Running active learning cycle...")
        try:
            predictions_file, candidates_file = run_active_learning_for_class(
                positive_class_name=class_name,
                model_path=current_model,
                run_number=cycle,
                selection_mode=selection_mode,
            )
            print(f"   ✅ Generated: {candidates_file}")
        except Exception as e:
            print(f"   ❌ Error in active learning: {e}")
            break

        # Step 2: Human labeling (simulated or manual)
        if simulate_human:
            labeled_count = simulate_human_labeling(candidates_file)
            if labeled_count == 0:
                print("   ⚠️ No samples were labeled, ending workflow")
                break
        else:
            print(f"   👤 MANUAL STEP: Please open {candidates_file}")
            print("      Fill in the 'needs_human_label' column with 0 or 1")
            input("      Press Enter when labeling is complete...")

        # Step 3: Merge human labels into training set
        print("Step 2: Merging human labels...")
        try:
            # Find current training set
            current_training_set = f"training_sets/training_set_v{cycle}.csv"
            if not os.path.exists(current_training_set):
                current_training_set = "training_sets/training_set_v1.csv"

            new_training_set = merge_training_sets(current_training_set, candidates_file)
            training_sets.append(new_training_set)
            print(f"   ✅ Created: {new_training_set}")

        except Exception as e:
            print(f"   ❌ Error merging labels: {e}")
            break

        # Step 4: Next cycle preparation
        if cycle < num_cycles:
            next_model = f"outputs/model_v{cycle + 1}.pt"
            print(f"Step 3: Next cycle will use model: {next_model}")
            print(f"        Train it using: {new_training_set}")

            if not os.path.exists(next_model):
                print(f"   ⚠️ Model {next_model} not found")
                print(
                    f"   💡 Train it with: python -m audioloop.train {new_training_set} -v {cycle + 1}"
                )
                if not simulate_human:
                    input("   Press Enter when model training is complete...")

            current_model = next_model

    print("\n🎉 Workflow Complete!")
    print(f"Generated training sets: {training_sets}")
    return training_sets


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate complete active learning workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 2 cycles with simulated human labeling
  python example_workflow.py --class-name siren --cycles 2

  # Run 3 cycles with manual human labeling
  python example_workflow.py --class-name dog_bark --cycles 3 --no-simulate

  # Use custom initial model
  python example_workflow.py --class-name gun_shot --model outputs/custom_model.pt

  # Use basic transition strategy
  python example_workflow.py --class-name siren --cycles 2 --selection-mode basic_transition

  # Use entropy-based selection
  python example_workflow.py --class-name dog_bark --cycles 3 --selection-mode entropy
        """,
    )

    parser.add_argument(
        "--class-name", required=True, help="Target class name (e.g., siren, dog_bark, gun_shot)"
    )
    parser.add_argument(
        "--model", default=None, help="Initial model path (default: outputs/model_v1.pt)"
    )
    parser.add_argument(
        "--cycles", type=int, default=2, help="Number of active learning cycles (default: 2)"
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="Disable simulated human labeling (requires manual input)",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["confidence", "entropy", "basic_transition"],
        default="confidence",
        help="Selection strategy (default: confidence)",
    )

    args = parser.parse_args()

    # Set default model if not specified
    if args.model is None:
        args.model = "outputs/model_v1.pt"

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("💡 Train an initial model first:")
        print("   python -m audioloop.train training_sets/training_set_v1.csv -v 1")
        return 1

    # Run the workflow
    try:
        training_sets = run_complete_workflow(
            class_name=args.class_name,
            initial_model=args.model,
            num_cycles=args.cycles,
            simulate_human=not args.no_simulate,
            selection_mode=args.selection_mode,
        )

        print("\n📊 Summary:")
        print(f"   Class: {args.class_name}")
        print(f"   Cycles completed: {len(training_sets)}")
        print(f"   Final training set: {training_sets[-1] if training_sets else 'None'}")

        print("\n💡 For convenience, consider using:")
        print(
            f"   python automated_workflow.py --class-name {args.class_name} --cycles {args.cycles} --selection-mode {args.selection_mode} --auto-label"
        )

        return 0

    except KeyboardInterrupt:
        print("\n⏹️ Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
