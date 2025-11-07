"""
Cycle-level stopping criteria for active learning loops.

This module provides stopping logic based on candidate metrics trends across
multiple active learning cycles, enabling automatic termination when the model
has learned effectively from the selected samples.
"""

import logging
import statistics

logger = logging.getLogger(__name__)


class CycleStoppingCriterion:
    """
    Base class for cross-cycle stopping criteria.

    Tracks candidate metrics across cycles and determines when to stop
    the active learning loop based on performance trends.
    """

    def __init__(self, config, metrics_history):
        """
        Initialize stopping criterion.

        Args:
            config: AudioLoopConfig with cycle stopping parameters
            metrics_history: Dict mapping cycle number to metrics dict
                            (from load_candidate_metrics_history())
        """
        self.config = config
        self.metrics_history = metrics_history
        self.best_cycle = 1
        self.best_rolling_avg = 0.0
        self.cycles_without_improvement = 0

    def should_stop(self, current_cycle):
        """
        Check if stopping criterion is met.

        Args:
            current_cycle: Current cycle number

        Returns:
            True if should stop, False otherwise

        Note: Subclasses must override this method.
        """
        raise NotImplementedError("Subclass must implement should_stop()")

    def get_best_cycle(self):
        """Get the cycle number with the best rolling average performance."""
        return self.best_cycle

    def get_status_message(self, current_cycle):
        """
        Get human-readable status message for display.

        Args:
            current_cycle: Current cycle number

        Returns:
            Status string describing current stopping state
        """
        return (
            f"Best cycle: {self.best_cycle} "
            f"(rolling avg: {self.best_rolling_avg:.3f}), "
            f"cycles without improvement: {self.cycles_without_improvement}"
        )

    def _calculate_rolling_avg(self, metric_key, window, end_cycle):
        """
        Calculate rolling average of a metric over the last N cycles.

        Args:
            metric_key: Metric to average (e.g., 'f1_score', 'recall')
            window: Number of cycles to average over
            end_cycle: Last cycle to include in calculation

        Returns:
            Rolling average value, or 0.0 if insufficient data
        """
        # Collect metrics from the most recent cycles that exist
        values = []
        for cycle in range(end_cycle, 0, -1):
            if len(values) >= window:
                break
            if cycle in self.metrics_history:
                try:
                    values.append(self.metrics_history[cycle][metric_key])
                except KeyError:
                    logger.warning(
                        f"Metric '{metric_key}' not found for cycle {cycle}, skipping"
                    )

        if len(values) < window:
            # Not enough data for full window
            return 0.0

        return statistics.mean(values)

    def _calculate_rolling_std(self, metric_key, window, end_cycle):
        """
        Calculate rolling standard deviation of a metric.

        Args:
            metric_key: Metric to calculate std for
            window: Number of cycles to use
            end_cycle: Last cycle to include

        Returns:
            Rolling standard deviation, or float('inf') if insufficient data
        """
        # Collect metrics from the most recent cycles that exist
        values = []
        for cycle in range(end_cycle, 0, -1):
            if len(values) >= window:
                break
            if cycle in self.metrics_history:
                try:
                    values.append(self.metrics_history[cycle][metric_key])
                except KeyError:
                    logger.warning(
                        f"Metric '{metric_key}' not found for cycle {cycle}, skipping"
                    )

        if len(values) < 2:
            # Need at least 2 values for std
            return float("inf")

        return statistics.stdev(values)

    def _has_improved(self, current_avg, best_avg):
        """
        Check if current average has improved beyond min_delta threshold.

        Args:
            current_avg: Current rolling average
            best_avg: Best rolling average seen so far

        Returns:
            True if current_avg > best_avg + min_delta
        """
        return current_avg > (best_avg + self.config.cycle_min_delta)


class LabelModeStoppingCriterion(CycleStoppingCriterion):
    """
    Stopping criterion for auto-labeling mode.

    Optimizes F1 score (balanced precision/recall) and stops when:
    - Minimum cycles completed
    - F1 rolling average stops improving (patience exhausted)
    - Performance is stable (low variance)
    """

    def should_stop(self, current_cycle):
        """
        Check if should stop based on F1 score trends.

        Args:
            current_cycle: Current cycle number

        Returns:
            True if all stopping conditions met, False otherwise
        """
        # Condition 1: Check minimum cycles
        if current_cycle < self.config.cycle_min_cycles:
            return False

        # Condition 2: Check we have enough history for rolling window
        cycles_available = sum(1 for c in range(1, current_cycle + 1) if c in self.metrics_history)
        if cycles_available < self.config.cycle_window:
            return False

        # Calculate rolling average F1
        current_rolling_avg = self._calculate_rolling_avg(
            "f1_score", self.config.cycle_window, current_cycle
        )

        if current_rolling_avg == 0.0:
            # Insufficient data
            return False

        # Check if improved beyond min_delta
        if self._has_improved(current_rolling_avg, self.best_rolling_avg):
            self.best_rolling_avg = current_rolling_avg
            self.best_cycle = current_cycle
            self.cycles_without_improvement = 0
            return False

        # Increment patience counter
        self.cycles_without_improvement += 1

        # Condition 3: Check patience exhausted
        if self.cycles_without_improvement < self.config.cycle_patience:
            return False

        # Condition 4: Check stability
        rolling_std = self._calculate_rolling_std(
            "f1_score", self.config.cycle_window, current_cycle
        )

        if rolling_std >= self.config.cycle_std_threshold:
            return False  # Not stable yet

        # All conditions met - stop!
        return True


class SearchModeStoppingCriterion(CycleStoppingCriterion):
    """
    Stopping criterion for high-recall search mode.

    Optimizes recall while maintaining precision above a floor, and stops when:
    - Minimum cycles completed
    - Recall rolling average stops improving (patience exhausted)
    - Performance is stable (low variance)
    - Precision remains above floor
    """

    def __init__(self, config, metrics_history):
        """Initialize with precision floor calculation."""
        super().__init__(config, metrics_history)
        self._precision_floor = self._calculate_precision_floor()

    def _calculate_precision_floor(self):
        """
        Calculate precision floor based on initial performance.

        Returns:
            Precision floor value (float)
        """
        if isinstance(self.config.precision_floor, (int, float)):
            # Manual precision floor
            return float(self.config.precision_floor)

        # Auto precision floor: max(0.30, initial_precision - 0.1)
        if 1 in self.metrics_history:
            try:
                initial_precision = self.metrics_history[1]["precision"]
                return max(0.30, initial_precision - 0.1)
            except KeyError:
                logger.warning("Precision not found for cycle 1, using fallback floor 0.30")

        # Fallback if cycle 1 not available
        logger.warning("Cycle 1 metrics not available, using fallback precision floor 0.30")
        return 0.30

    def should_stop(self, current_cycle):
        """
        Check if should stop based on recall trends and precision constraint.

        Args:
            current_cycle: Current cycle number

        Returns:
            True if all stopping conditions met, False otherwise
        """
        # Condition 1: Check minimum cycles
        if current_cycle < self.config.cycle_min_cycles:
            return False

        # Condition 2: Check we have enough history for rolling window
        cycles_available = sum(1 for c in range(1, current_cycle + 1) if c in self.metrics_history)
        if cycles_available < self.config.cycle_window:
            return False

        # Calculate rolling average recall
        current_rolling_avg = self._calculate_rolling_avg(
            "recall", self.config.cycle_window, current_cycle
        )

        if current_rolling_avg == 0.0:
            # Insufficient data
            return False

        # Check if improved beyond min_delta
        if self._has_improved(current_rolling_avg, self.best_rolling_avg):
            self.best_rolling_avg = current_rolling_avg
            self.best_cycle = current_cycle
            self.cycles_without_improvement = 0
            return False

        # Increment patience counter
        self.cycles_without_improvement += 1

        # Condition 3: Check patience exhausted
        if self.cycles_without_improvement < self.config.cycle_patience:
            return False

        # Condition 4: Check precision floor constraint
        rolling_precision = self._calculate_rolling_avg(
            "precision", self.config.cycle_window, current_cycle
        )

        if rolling_precision < self._precision_floor:
            # Precision below floor - cannot stop yet
            return False

        # Condition 5: Check stability (more tolerant threshold for search mode)
        # Use 0.10 instead of config value for search mode
        rolling_std = self._calculate_rolling_std(
            "recall", self.config.cycle_window, current_cycle
        )

        if rolling_std >= 0.10:
            return False  # Not stable yet

        # All conditions met - stop!
        return True


def create_cycle_stopping_criterion(config, metrics_history):
    """
    Create a stopping criterion based on configuration.

    Args:
        config: AudioLoopConfig with cycle_stopping_strategy setting
        metrics_history: Dict mapping cycle number to metrics dict

    Returns:
        CycleStoppingCriterion instance, or None if strategy is "none"

    Raises:
        ValueError: If strategy is invalid
    """
    strategy = config.cycle_stopping_strategy

    if strategy == "label":
        return LabelModeStoppingCriterion(config, metrics_history)
    elif strategy == "search":
        return SearchModeStoppingCriterion(config, metrics_history)
    elif strategy == "none":
        return None
    else:
        raise ValueError(
            f"Unknown cycle stopping strategy: '{strategy}'. "
            f"Expected: 'label', 'search', or 'none'"
        )
