#!/usr/bin/env python3
"""
Stopping Criteria Demo Script

This script demonstrates how to use different stopping criteria in AudioLoop
for controlling when model training should stop.
"""

import random
import sys
import time
from pathlib import Path

# Add the audioloop directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audioloop.utils.stopping_criteria import (
    AccuracyCriterion,
    PlateauCriterion,
    TrainingStoppingCriterion,
)


def simulate_training_metrics(
    epochs: int, pattern: str = "improving"
) -> list[tuple[int, float, float]]:
    """
    Simulate training metrics for demonstration purposes.

    Args:
        epochs: Number of epochs to simulate
        pattern: Training pattern ('improving', 'plateau', 'oscillating')

    Returns:
        List of (epoch, accuracy, loss) tuples
    """
    metrics = []
    random.seed(42)  # For reproducible results

    for epoch in range(epochs):
        if pattern == "improving":
            # Steady improvement with some noise
            base_accuracy = min(0.5 + (epoch * 0.01), 0.99)
            base_loss = max(0.8 - (epoch * 0.015), 0.05)

            # Add some noise
            accuracy = base_accuracy + random.uniform(-0.02, 0.02)
            loss = base_loss + random.uniform(-0.02, 0.02)

        elif pattern == "plateau":
            # Initial improvement, then plateau
            if epoch < 20:
                base_accuracy = 0.5 + (epoch * 0.02)
                base_loss = 0.8 - (epoch * 0.03)
            else:
                base_accuracy = 0.85
                base_loss = 0.2

            accuracy = base_accuracy + random.uniform(-0.01, 0.01)
            loss = base_loss + random.uniform(-0.01, 0.01)

        elif pattern == "oscillating":
            # Oscillating improvement
            import math

            base_accuracy = 0.6 + (epoch * 0.005)
            base_loss = 0.6 - (epoch * 0.008)

            accuracy = base_accuracy + 0.05 * math.sin(epoch * 0.5)
            loss = base_loss + 0.03 * math.cos(epoch * 0.3)

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        # Clamp values to realistic ranges
        accuracy = max(0.0, min(1.0, accuracy))
        loss = max(0.01, loss)

        metrics.append((epoch, accuracy, loss))

    return metrics


def run_training_simulation(
    criterion: TrainingStoppingCriterion,
    pattern: str = "improving",
    max_epochs: int = 1000,
    verbose: bool = True,
) -> tuple[int, list[tuple[int, float, float]]]:
    """
    Simulate a training run with the given stopping criterion.

    Args:
        criterion: Stopping criterion to use
        pattern: Training pattern to simulate
        max_epochs: Maximum epochs to simulate
        verbose: Whether to print progress

    Returns:
        Tuple of (stop_epoch, metrics_history)
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training Simulation: {criterion.__class__.__name__}")
        print(f"Pattern: {pattern}")
        print(f"{'=' * 60}")

    # Reset criterion state
    criterion.reset()

    # Generate training metrics
    all_metrics = simulate_training_metrics(max_epochs, pattern)

    # Run through training loop
    history = []

    for epoch, accuracy, loss in all_metrics:
        history.append((epoch, accuracy, loss))

        # Print progress every 10 epochs
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Accuracy={accuracy:.3f}, Loss={loss:.3f}")

        # Check stopping criterion
        if criterion.should_stop(epoch, accuracy, loss):
            if verbose:
                print(f"\n>>> STOPPED at epoch {epoch}")
                print(f"    Final accuracy: {accuracy:.3f}")
                print(f"    Final loss: {loss:.3f}")

                # Print criterion-specific info
                if hasattr(criterion, "best_train_loss"):
                    best_loss = getattr(criterion, "best_train_loss", None)
                    if best_loss is not None:
                        print(f"    Best loss: {best_loss:.3f}")
                if hasattr(criterion, "epochs_without_improvement"):
                    epochs_without_improvement = getattr(
                        criterion, "epochs_without_improvement", None
                    )
                    if epochs_without_improvement is not None:
                        print(f"    Epochs without improvement: {epochs_without_improvement}")

            return epoch, history

    if verbose:
        print(f"\n>>> Reached maximum epochs ({max_epochs})")

    return max_epochs - 1, history


def demo_accuracy_criterion():
    """Demonstrate AccuracyCriterion usage."""
    print("\n" + "=" * 80)
    print("DEMO 1: AccuracyCriterion")
    print("=" * 80)

    # Default criterion
    criterion = AccuracyCriterion()
    print(f"Default AccuracyCriterion: max_epochs={criterion.max_epochs}")

    stop_epoch, _ = run_training_simulation(criterion, "improving", max_epochs=100)
    print(f"Stopped at epoch: {stop_epoch}")

    # Custom criterion
    print("\n" + "-" * 40)
    criterion = AccuracyCriterion(max_epochs=50)
    print(f"Custom AccuracyCriterion: max_epochs={criterion.max_epochs}")

    stop_epoch, _ = run_training_simulation(criterion, "improving", max_epochs=100)
    print(f"Stopped at epoch: {stop_epoch}")


def demo_plateau_criterion():
    """Demonstrate PlateauCriterion usage."""
    print("\n" + "=" * 80)
    print("DEMO 2: PlateauCriterion")
    print("=" * 80)

    # Default criterion with plateau pattern
    criterion = PlateauCriterion()
    print(
        f"Default PlateauCriterion: patience={criterion.patience}, "
        f"min_delta={criterion.min_delta}, max_epochs={criterion.max_epochs}"
    )

    stop_epoch, _ = run_training_simulation(criterion, "plateau", max_epochs=100)
    print(f"Stopped at epoch: {stop_epoch}")

    # Custom criterion with different parameters
    print("\n" + "-" * 40)
    criterion = PlateauCriterion(patience=10, min_delta=0.005, max_epochs=200)
    print(
        f"Custom PlateauCriterion: patience={criterion.patience}, "
        f"min_delta={criterion.min_delta}, max_epochs={criterion.max_epochs}"
    )

    stop_epoch, _ = run_training_simulation(criterion, "plateau", max_epochs=100)
    print(f"Stopped at epoch: {stop_epoch}")


def demo_custom_criterion():
    """Demonstrate custom stopping criterion."""
    print("\n" + "=" * 80)
    print("DEMO 3: Custom Stopping Criterion")
    print("=" * 80)

    class TargetAccuracyCriterion(TrainingStoppingCriterion):
        """Stop when target accuracy is reached."""

        def __init__(self, target_accuracy: float = 0.9, max_epochs: int = 1000):
            self.target_accuracy = target_accuracy
            self.max_epochs = max_epochs

        def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
            return train_accuracy >= self.target_accuracy or epoch >= self.max_epochs - 1

    criterion = TargetAccuracyCriterion(target_accuracy=0.8, max_epochs=200)
    print(
        f"TargetAccuracyCriterion: target_accuracy={criterion.target_accuracy}, "
        f"max_epochs={criterion.max_epochs}"
    )

    stop_epoch, _ = run_training_simulation(criterion, "improving", max_epochs=100)
    print(f"Stopped at epoch: {stop_epoch}")


def demo_combined_criteria():
    """Demonstrate combining multiple criteria."""
    print("\n" + "=" * 80)
    print("DEMO 4: Combined Stopping Criteria")
    print("=" * 80)

    class CombinedCriterion(TrainingStoppingCriterion):
        """Combine multiple stopping criteria."""

        def __init__(self, criteria: list[TrainingStoppingCriterion]):
            self.criteria = criteria

        def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
            # Stop if ANY criterion says to stop
            for criterion in self.criteria:
                if criterion.should_stop(epoch, train_accuracy, train_loss, val_accuracy, val_loss):
                    return True
            return False

        def reset(self):
            for criterion in self.criteria:
                criterion.reset()

    # Combine accuracy and plateau criteria
    combined = CombinedCriterion(
        [
            AccuracyCriterion(max_epochs=200),
            PlateauCriterion(patience=15, min_delta=0.01, max_epochs=200),
        ]
    )

    print("CombinedCriterion: AccuracyCriterion + PlateauCriterion")
    stop_epoch, _ = run_training_simulation(combined, "plateau", max_epochs=100)
    print(f"Stopped at epoch: {stop_epoch}")


def demo_performance_comparison():
    """Compare performance of different criteria."""
    print("\n" + "=" * 80)
    print("DEMO 5: Performance Comparison")
    print("=" * 80)

    criteria = [
        ("AccuracyCriterion", AccuracyCriterion(max_epochs=100)),
        ("PlateauCriterion", PlateauCriterion(patience=20, max_epochs=100)),
        (
            "PlateauCriterion (sensitive)",
            PlateauCriterion(patience=5, min_delta=0.001, max_epochs=100),
        ),
    ]

    patterns = ["improving", "plateau", "oscillating"]

    print(f"{'Criterion':<25} {'Pattern':<12} {'Stop Epoch':<10} {'Time (ms)':<10}")
    print("-" * 60)

    for pattern in patterns:
        for name, criterion in criteria:
            start_time = time.perf_counter()
            stop_epoch, _ = run_training_simulation(
                criterion, pattern, max_epochs=100, verbose=False
            )
            end_time = time.perf_counter()

            duration_ms = (end_time - start_time) * 1000
            print(f"{name:<25} {pattern:<12} {stop_epoch:<10} {duration_ms:<10.2f}")


def main():
    """Run all stopping criteria demonstrations."""
    print("AudioLoop Stopping Criteria Demo")
    print("=" * 80)
    print("This script demonstrates various stopping criteria available in AudioLoop.")
    print("Each demo shows how different criteria behave with simulated training data.")

    try:
        demo_accuracy_criterion()
        demo_plateau_criterion()
        demo_custom_criterion()
        demo_combined_criteria()
        demo_performance_comparison()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("Key Takeaways:")
        print("- AccuracyCriterion: Simple, stops at perfect accuracy or max epochs")
        print("- PlateauCriterion: Sophisticated, stops when loss plateaus")
        print("- Custom criteria: Easy to implement for specific needs")
        print("- Combined criteria: Mix multiple strategies for robust stopping")
        print("- Performance: All criteria are lightweight and fast")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
        raise


if __name__ == "__main__":
    main()
