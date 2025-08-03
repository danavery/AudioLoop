"""
Utility functions for candidate selection.

This module provides helper functions for loading predictions, saving candidates,
and printing selection statistics used across all candidate selection strategies.
"""

import csv
import os
from typing import Any


def load_predictions(predictions_file: str) -> list[dict[str, Any]]:
    """Load predictions from CSV file."""
    predictions = []
    with open(predictions_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["entropy"] = float(row["entropy"])
            predictions.append(row)
    return predictions


def save_candidates(candidates: list[dict[str, Any]], candidates_csv: str) -> None:
    """Save candidates to CSV file."""
    if not candidates:
        print("No candidates to save.")
        return

    # Add labeling helper columns
    for candidate in candidates:
        candidate["needs_human_label"] = ""  # Empty column for human to fill
        candidate["human_confidence"] = ""  # Human confidence in the label

    # Ensure outputs directory exists
    os.makedirs(
        os.path.dirname(candidates_csv) if os.path.dirname(candidates_csv) else "outputs",
        exist_ok=True,
    )

    # Save candidates
    fieldnames = list(candidates[0].keys())
    with open(candidates_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    print(f"Candidates saved to: {candidates_csv}")


def print_selection_statistics(
    all_predictions: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    strategy_name: str,
    positive_class_name: str = "positive",
    negative_class_name: str = "negative",
    min_confidence: float = 0.8,
) -> None:
    """Print comprehensive statistics about the selection process."""
    total_samples = len(all_predictions)

    if total_samples == 0:
        print("No predictions available.")
        return

    # Calculate overall accuracy statistics with robust boolean conversion
    correct_predictions = 0
    for p in all_predictions:
        correct_val = p["correct"]
        if isinstance(correct_val, str):
            is_correct = correct_val.lower() in ("true", "1")
        elif isinstance(correct_val, bool):
            is_correct = correct_val
        elif isinstance(correct_val, int | float):
            is_correct = bool(correct_val)
        else:
            is_correct = False

        if is_correct:
            correct_predictions += 1

    overall_accuracy = correct_predictions / total_samples

    # Calculate accuracy by true class
    true_positive_samples = []
    true_negative_samples = []

    for p in all_predictions:
        true_is_positive = p["true_is_positive"]
        # Handle string/boolean values with robust conversion
        if isinstance(true_is_positive, str):
            true_is_positive = true_is_positive.lower() in ("true", "1")
        elif isinstance(true_is_positive, int | float):
            true_is_positive = bool(true_is_positive)
        else:
            true_is_positive = bool(true_is_positive)

        if true_is_positive:
            true_positive_samples.append(p)
        else:
            true_negative_samples.append(p)

    # Calculate correct predictions with robust boolean conversion
    def is_prediction_correct(pred):
        correct_val = pred["correct"]
        if isinstance(correct_val, str):
            return correct_val.lower() in ("true", "1")
        if isinstance(correct_val, bool):
            return correct_val
        if isinstance(correct_val, int | float):
            return bool(correct_val)
        return False

    positive_correct = sum(1 for p in true_positive_samples if is_prediction_correct(p))
    negative_correct = sum(1 for p in true_negative_samples if is_prediction_correct(p))

    positive_accuracy = (
        positive_correct / len(true_positive_samples) if true_positive_samples else 0
    )
    negative_accuracy = (
        negative_correct / len(true_negative_samples) if true_negative_samples else 0
    )

    # Overall confidence statistics
    all_confidences = [p["confidence"] for p in all_predictions]
    high_conf_count = sum(1 for conf in all_confidences if conf >= min_confidence)
    high_conf_percentage = high_conf_count / total_samples

    avg_confidence = sum(all_confidences) / len(all_confidences)
    min_conf = min(all_confidences)
    max_conf = max(all_confidences)

    # Sanity check for accuracy calculations
    expected_overall_accuracy = (positive_correct + negative_correct) / total_samples
    accuracy_mismatch = abs(overall_accuracy - expected_overall_accuracy) > 0.001

    print("\nModel Performance Summary:")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(
        f"True {positive_class_name} Accuracy: {positive_accuracy:.3f} ({positive_correct}/{len(true_positive_samples)})"
    )
    print(
        f"True {negative_class_name} Accuracy: {negative_accuracy:.3f} ({negative_correct}/{len(true_negative_samples)})"
    )

    if accuracy_mismatch:
        print("⚠️  WARNING: Accuracy calculation mismatch detected!")
        print(f"   Expected overall accuracy: {expected_overall_accuracy:.3f}")
        print(f"   Calculated overall accuracy: {overall_accuracy:.3f}")

    print(f"Overall Confidence: avg={avg_confidence:.3f}, range={min_conf:.3f}-{max_conf:.3f}")
    print(
        f"High confidence samples (≥{min_confidence}): {high_conf_count}/{total_samples} ({high_conf_percentage:.1%})"
    )

    # Show current selection mode and recommendations
    print(f"Selection strategy: {strategy_name}")
    if overall_accuracy > 0.95 and high_conf_percentage > 0.8:
        if "confidence" in strategy_name.lower():
            print("💡 Recommendation: Model is highly accurate and confident")
            print("   Consider switching to entropy-based selection for uncertainty sampling")
            print("   Current confidence-based selection may not find challenging cases")
        else:
            print("💡 Model is highly accurate and confident")
            print("   Using entropy-based selection should help find challenging cases")

    # Selection statistics
    positive_candidates = [c for c in selected_candidates if c["prediction"] == positive_class_name]
    negative_candidates = [c for c in selected_candidates if c["prediction"] == negative_class_name]

    positive_preds = [p for p in all_predictions if p["prediction"] == positive_class_name]
    negative_preds = [p for p in all_predictions if p["prediction"] == negative_class_name]

    print("\nActive Learning Candidate Selection:")
    print(f"Available {positive_class_name} predictions: {len(positive_preds)}")
    print(f"Available {negative_class_name} predictions: {len(negative_preds)}")
    print(f"Selected {len(positive_candidates)} {positive_class_name} candidates")
    print(f"Selected {len(negative_candidates)} {negative_class_name} candidates")
    print(f"Total candidates for labeling: {len(selected_candidates)}")

    # Print candidate summaries
    if positive_candidates:
        conf_values = [p["confidence"] for p in positive_candidates]
        print(
            f"{positive_class_name} confidence range: {min(conf_values):.3f} - {max(conf_values):.3f}"
        )
        print(
            f"  First 3 {positive_class_name} samples: {[p['filename'] for p in positive_candidates[:3]]}"
        )
    if negative_candidates:
        conf_values = [p["confidence"] for p in negative_candidates]
        print(
            f"{negative_class_name} confidence range: {min(conf_values):.3f} - {max(conf_values):.3f}"
        )
        print(
            f"  First 3 {negative_class_name} samples: {[p['filename'] for p in negative_candidates[:3]]}"
        )


# Factory function
def create_strategy(selection_mode: str, **kwargs):
    """Create a selection strategy from a mode string."""
    from .basic_transition import BasicTransitionStrategy
    from .confidence import ConfidenceStrategy
    from .entropy import EntropyStrategy
    from .stratified import StratifiedUncertaintyStrategy

    strategies = {
        "confidence": ConfidenceStrategy,
        "entropy": EntropyStrategy,
        "basic_transition": BasicTransitionStrategy,
        "stratified_uncertainty": StratifiedUncertaintyStrategy,
    }

    if selection_mode not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown selection mode: '{selection_mode}'. Available: {available}")

    return strategies[selection_mode](**kwargs)