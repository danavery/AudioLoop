"""
Candidate selection strategies for active learning.

This module provides a pluggable architecture for candidate selection,
following the same pattern as stopping_criteria.py.
"""

import csv
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from .metrics_utils import calculate_binary_metrics


class CandidateSelectionStrategy(ABC):
    """Base class for candidate selection strategies."""

    @abstractmethod
    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Select candidates from predictions for human labeling.

        Args:
            predictions: List of prediction dictionaries
            num_candidates: Total number of candidates to select
            positive_class_name: Name for positive class
            negative_class_name: Name for negative class
            **kwargs: Strategy-specific parameters

        Returns:
            List of selected candidates
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """Return strategy name for logging."""
        return self.__class__.__name__


class ConfidenceStrategy(CandidateSelectionStrategy):
    """Select candidates with highest confidence scores."""

    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select high-confidence candidates using existing algorithm."""
        positive_percentage = kwargs.get("positive_percentage", 0.8)
        candidate_pool_multiplier = kwargs.get("candidate_pool_multiplier", 5)
        random_seed = kwargs.get("random_seed", int(time.time()))

        # Calculate numbers based on percentage
        num_positive = int(num_candidates * positive_percentage)
        num_negative = num_candidates - num_positive

        # Separate positive and negative predictions
        positive_preds = [p for p in predictions if p["prediction"] == positive_class_name]
        negative_preds = [p for p in predictions if p["prediction"] == negative_class_name]

        # Sort by confidence (highest first)
        positive_preds.sort(key=lambda x: x["confidence"], reverse=True)
        negative_preds.sort(key=lambda x: x["confidence"], reverse=True)

        # Create broader candidate pools for sampling
        positive_pool_size = min(len(positive_preds), num_positive * candidate_pool_multiplier)
        negative_pool_size = min(len(negative_preds), num_negative * candidate_pool_multiplier)

        positive_pool = positive_preds[:positive_pool_size]
        negative_pool = negative_preds[:negative_pool_size]

        # Randomly sample from the pools to improve diversity
        random.seed(random_seed)

        positive_candidates = random.sample(positive_pool, min(num_positive, len(positive_pool)))
        negative_candidates = random.sample(negative_pool, min(num_negative, len(negative_pool)))

        # Combine and shuffle final candidates
        all_candidates = positive_candidates + negative_candidates
        random.shuffle(all_candidates)

        return all_candidates

    def get_name(self) -> str:
        return "Confidence-Based"


class EntropyStrategy(CandidateSelectionStrategy):
    """Select candidates with highest entropy (uncertainty sampling)."""

    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select high-entropy candidates using existing algorithm."""
        positive_percentage = kwargs.get("positive_percentage", 0.8)
        candidate_pool_multiplier = kwargs.get("candidate_pool_multiplier", 5)
        random_seed = kwargs.get("random_seed", int(time.time()))

        # Calculate numbers based on percentage
        num_positive = int(num_candidates * positive_percentage)
        num_negative = num_candidates - num_positive

        # Separate positive and negative predictions
        positive_preds = [p for p in predictions if p["prediction"] == positive_class_name]
        negative_preds = [p for p in predictions if p["prediction"] == negative_class_name]

        # Sort by entropy (highest first = most uncertain)
        positive_preds.sort(key=lambda x: x["entropy"], reverse=True)
        negative_preds.sort(key=lambda x: x["entropy"], reverse=True)

        # Create broader candidate pools for sampling
        positive_pool_size = min(len(positive_preds), num_positive * candidate_pool_multiplier)
        negative_pool_size = min(len(negative_preds), num_negative * candidate_pool_multiplier)

        positive_pool = positive_preds[:positive_pool_size]
        negative_pool = negative_preds[:negative_pool_size]

        # Randomly sample from the pools to improve diversity
        random.seed(random_seed)

        positive_candidates = random.sample(positive_pool, min(num_positive, len(positive_pool)))
        negative_candidates = random.sample(negative_pool, min(num_negative, len(negative_pool)))

        # Combine and shuffle final candidates
        all_candidates = positive_candidates + negative_candidates
        random.shuffle(all_candidates)

        return all_candidates

    def get_name(self) -> str:
        return "Entropy-Based (Uncertainty Sampling)"


class BasicTransitionStrategy(CandidateSelectionStrategy):
    """
    Basic transition strategy that switches from confidence to entropy
    based on model performance metrics (F1, confidence, variance thresholds).
    """

    def __init__(
        self,
        f1_threshold: float | None = None,
        confidence_threshold: float | None = None,
        variance_threshold: float | None = None,
        auto_thresholds: bool = False,
        estimated_positive_pct: float = 0.05,
        training_set_size: int | None = None,
    ):
        # Auto-calculate thresholds if requested or if any threshold is None
        if auto_thresholds or any(
            t is None for t in [f1_threshold, confidence_threshold, variance_threshold]
        ):
            from .adaptive_thresholds import threshold_calculator

            calc_f1, calc_conf, calc_var = threshold_calculator.calculate_thresholds(
                estimated_positive_pct, training_set_size
            )

            self.f1_threshold = f1_threshold if f1_threshold is not None else calc_f1
            self.confidence_threshold = (
                confidence_threshold if confidence_threshold is not None else calc_conf
            )
            self.variance_threshold = (
                variance_threshold if variance_threshold is not None else calc_var
            )

            # Print analysis if fully auto
            if auto_thresholds:
                threshold_calculator.print_analysis(
                    estimated_positive_pct,
                    self.f1_threshold,
                    self.confidence_threshold,
                    self.variance_threshold,
                    training_set_size,
                )
        else:
            # Use provided thresholds or defaults
            self.f1_threshold = f1_threshold if f1_threshold is not None else 0.2
            self.confidence_threshold = (
                confidence_threshold if confidence_threshold is not None else 0.9
            )
            self.variance_threshold = variance_threshold if variance_threshold is not None else 0.12

        # Delegate to existing strategies
        self.confidence_strategy = ConfidenceStrategy()
        self.entropy_strategy = EntropyStrategy()

        # Track which strategy we're using (for reporting)
        self.current_strategy_name = "confidence"

    def _should_transition(self, metrics: dict) -> bool:
        """
        Determine if we should transition from confidence to entropy selection.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            bool: True if all transition criteria are met
        """
        if not metrics:
            return False

        f1_met = metrics.get("f1_score", 0) > self.f1_threshold
        confidence_met = metrics.get("mean_confidence", 0) > self.confidence_threshold
        variance_met = metrics.get("std_confidence", 1) < self.variance_threshold

        return f1_met and confidence_met and variance_met

    def _print_transition_analysis(self, metrics: dict, should_transition: bool) -> None:
        """
        Print detailed transition analysis.

        Args:
            metrics: Dictionary of calculated metrics
            should_transition: Whether transition criteria are met
        """
        if not metrics:
            print("  Basic Transition: No metrics available, using confidence-based selection")
            return

        print("  Basic Transition Analysis:")

        f1_met = metrics.get("f1_score", 0) > self.f1_threshold
        confidence_met = metrics.get("mean_confidence", 0) > self.confidence_threshold
        variance_met = metrics.get("std_confidence", 1) < self.variance_threshold

        print(
            f"    {'✓' if f1_met else '✗'} F1 Score: {metrics.get('f1_score', 0):.3f} (>{self.f1_threshold} required)"
        )
        print(
            f"    {'✓' if confidence_met else '✗'} Mean Confidence: {metrics.get('mean_confidence', 0):.3f} (>{self.confidence_threshold} required)"
        )
        print(
            f"    {'✓' if variance_met else '✗'} Std Confidence: {metrics.get('std_confidence', 1):.3f} (<{self.variance_threshold} required)"
        )

        if should_transition:
            print("    → Using ENTROPY-based selection (uncertainty sampling)")
        else:
            print("    → Using CONFIDENCE-based selection")

    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select candidates using auto-transition logic."""

        # Calculate metrics using imported function
        metrics = calculate_binary_metrics(predictions)

        # Make transition decision
        should_transition = self._should_transition(metrics)

        # Print analysis
        self._print_transition_analysis(metrics, should_transition)

        # Choose strategy based on transition decision
        if should_transition:
            self.current_strategy_name = "entropy"
            strategy = self.entropy_strategy
        else:
            self.current_strategy_name = "confidence"
            strategy = self.confidence_strategy

        # Delegate to chosen strategy
        return strategy.select_candidates(
            predictions, num_candidates, positive_class_name, negative_class_name, **kwargs
        )

    def get_name(self) -> str:
        return f"Basic Transition ({self.current_strategy_name.title()}-Based)"


# Factory function
def create_strategy(selection_mode: str, **kwargs) -> CandidateSelectionStrategy:
    """Create a selection strategy from a mode string."""
    strategies = {
        "confidence": ConfidenceStrategy,
        "entropy": EntropyStrategy,
        "basic_transition": BasicTransitionStrategy,
    }

    if selection_mode not in strategies:
        available = ", ".join(strategies.keys())
        raise ValueError(f"Unknown selection mode: '{selection_mode}'. Available: {available}")

    return strategies[selection_mode](**kwargs)


# Utility functions
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
