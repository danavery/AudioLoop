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
        print(f"\n{'='*60}")
        print(f"Training Simulation: {criterion.__class__.__name__}")
        print(f"Pattern: {pattern}")
        print(f"{'='*60}")

    # Reset criterion state
    criterion.reset()

    # No dataset size needed for stopping criteria

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

                # Print criterion-specific info using polymorphic approach
                if isinstance(criterion, PlateauCriterion):
                    print(f"    Best loss: {criterion.best_train_loss:.3f}")
                    print(f"    Epochs without improvement: {criterion.epochs_without_improvement}")

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


def demo_plateau_with_accuracy_floor():
    """Demonstrate PlateauCriterion with accuracy_floor parameter."""
    print("\n" + "=" * 80)
    print("DEMO 3: PlateauCriterion with Accuracy Floor")
    print("=" * 80)

    # Default plateau criterion
    criterion = PlateauCriterion()
    print(f"Default PlateauCriterion: patience={criterion.patience}, accuracy_floor={criterion.accuracy_floor}")

    # Test with accuracy floor
    print("\n" + "-" * 40)
    print("Enhanced PlateauCriterion with accuracy floor (0.9)")
    enhanced_criterion = PlateauCriterion(patience=10, accuracy_floor=0.9)
    print(f"Parameters: patience={enhanced_criterion.patience}, accuracy_floor={enhanced_criterion.accuracy_floor}")
    stop_epoch, _ = run_training_simulation(enhanced_criterion, "plateau", max_epochs=100)
    print(f"Stopped at epoch {stop_epoch}")

    # Test without accuracy floor
    print("\n" + "-" * 40)
    print("Standard PlateauCriterion (no accuracy floor)")
    standard_criterion = PlateauCriterion(patience=10)
    stop_epoch, _ = run_training_simulation(standard_criterion, "plateau", max_epochs=100)
    print(f"Stopped at epoch {stop_epoch}")

    # Demonstrate accuracy floor behavior with step-by-step simulation
    print("\n" + "-" * 40)
    print("Step-by-step accuracy floor demonstration")
    demo_criterion = PlateauCriterion(patience=3, accuracy_floor=0.85)
    print(f"Criterion: patience={demo_criterion.patience}, accuracy_floor={demo_criterion.accuracy_floor}")

    # Simulate training sequence
    training_steps = [
        (0, 0.7, 1.0),   # Below floor
        (1, 0.8, 1.1),   # Below floor, loss worse
        (2, 0.9, 0.9),   # Above floor, loss better
        (3, 0.91, 1.0),  # Above floor, loss worse (patience=1)
        (4, 0.92, 1.1),  # Above floor, loss worse (patience=2)
        (5, 0.93, 1.2),  # Above floor, loss worse (patience=3, should stop)
    ]

    for epoch, accuracy, loss in training_steps:
        should_stop = demo_criterion.should_stop(epoch, accuracy, loss)
        print(f"  Epoch {epoch}: accuracy={accuracy:.2f}, loss={loss:.2f}, "
              f"patience={demo_criterion.epochs_without_improvement}, stop={should_stop}")
        if should_stop:
            break


def demo_custom_criterion():
    """Demonstrate custom stopping criterion."""
    print("\n" + "=" * 80)
    print("DEMO 4: Custom Stopping Criterion")
    print("=" * 80)

    class TargetAccuracyCriterion(TrainingStoppingCriterion):
        """Stop when target accuracy is reached."""

        def __init__(self, target_accuracy: float = 0.9, max_epochs: int = 1000):
            self.target_accuracy = target_accuracy
            self.max_epochs = max_epochs
            self.best_model_state = None

        def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
            return train_accuracy >= self.target_accuracy or epoch >= self.max_epochs - 1

        def should_update_best_model(self) -> bool:
            return False  # Simple criterion doesn't track best model

        def update_best_model(self, model_state) -> None:
            self.best_model_state = model_state

        def get_best_model_state(self):
            return self.best_model_state

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
    print("DEMO 5: Combined Stopping Criteria")
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

        def should_update_best_model(self) -> bool:
            # Update if any criterion says to update
            for criterion in self.criteria:
                if criterion.should_update_best_model():
                    return True
            return False

        def update_best_model(self, model_state) -> None:
            # Update all criteria
            for criterion in self.criteria:
                criterion.update_best_model(model_state)

        def get_best_model_state(self):
            # Return the first available best model state
            for criterion in self.criteria:
                state = criterion.get_best_model_state()
                if state is not None:
                    return state
            return None

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
    print("DEMO 6: Performance Comparison")
    print("=" * 80)

    criteria = [
        ("AccuracyCriterion", AccuracyCriterion(max_epochs=100)),
        ("PlateauCriterion", PlateauCriterion(patience=20, max_epochs=100)),
        (
            "PlateauCriterion (sensitive)",
            PlateauCriterion(patience=5, min_delta=0.001, max_epochs=100),
        ),
        ("PlateauCriterion (accuracy floor)", PlateauCriterion(patience=15, accuracy_floor=0.9, max_epochs=100)),
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
        demo_plateau_with_accuracy_floor()
        demo_custom_criterion()
        demo_combined_criteria()
        demo_performance_comparison()

        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        print("Key Takeaways:")
        print("- AccuracyCriterion: Simple, stops at perfect accuracy or max epochs")
        print("- PlateauCriterion: Sophisticated, stops when loss plateaus")
        print("- PlateauCriterion with accuracy_floor: Only patient when accuracy is high enough")
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
