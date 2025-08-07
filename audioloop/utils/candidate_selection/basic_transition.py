"""
Basic transition strategy that switches from confidence to entropy selection.

Automatically transitions between confidence-based and entropy-based selection
based on model performance metrics (F1, confidence, variance thresholds).
"""

from typing import Any

from audioloop.utils.metrics_utils import calculate_binary_metrics

from .base import CandidateSelectionStrategy
from .confidence import ConfidenceStrategy
from .entropy import EntropyStrategy


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
        # Validate threshold specification
        provided_thresholds = [f1_threshold, confidence_threshold, variance_threshold]
        num_provided = sum(1 for t in provided_thresholds if t is not None)

        if auto_thresholds and num_provided > 0:
            raise ValueError("Cannot specify both auto_thresholds=True and manual thresholds")
        if 0 < num_provided < 3:
            raise ValueError(
                "Must specify all three thresholds (f1_threshold, confidence_threshold, variance_threshold) "
                "or none of them. Use auto_thresholds=True for automatic calculation."
            )

        # Apply validated threshold configuration
        if auto_thresholds:
            # Auto-calculate all thresholds
            from audioloop.utils.adaptive_thresholds import threshold_calculator

            self.f1_threshold, self.confidence_threshold, self.variance_threshold = (
                threshold_calculator.calculate_thresholds(estimated_positive_pct, training_set_size)
            )

            # Print analysis for auto-calculated thresholds
            threshold_calculator.print_analysis(
                estimated_positive_pct,
                self.f1_threshold,
                self.confidence_threshold,
                self.variance_threshold,
                training_set_size,
            )
        elif num_provided == 3:
            # Use all provided thresholds
            self.f1_threshold = f1_threshold
            self.confidence_threshold = confidence_threshold
            self.variance_threshold = variance_threshold
        else:
            # Use hard-coded defaults (no manual thresholds provided)
            self.f1_threshold = 0.2
            self.confidence_threshold = 0.9
            self.variance_threshold = 0.12

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
        return "Basic Transition"

    def get_active_strategy_name(self) -> str:
        return f"Basic Transition ({self.current_strategy_name.title()}-Based)"
