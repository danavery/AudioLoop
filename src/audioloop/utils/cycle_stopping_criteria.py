"""
Cycle-level stopping criteria for active learning loops.

This module provides stopping logic based on candidate metrics trends across
multiple active learning cycles, enabling automatic termination when the model
has learned effectively from the selected samples.

The 'churn' criterion is the exception: instead of candidate-batch metrics it watches
prediction churn on the unlabeled pool, which needs no ground truth (see
ChurnStoppingCriterion).
"""

import csv
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
                    logger.warning(f"Metric '{metric_key}' not found for cycle {cycle}, skipping")

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
                    logger.warning(f"Metric '{metric_key}' not found for cycle {cycle}, skipping")

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

        # All conditions met - stop!
        return rolling_std < self.config.cycle_std_threshold


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
        if isinstance(self.config.precision_floor, int | float):
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
        rolling_std = self._calculate_rolling_std("recall", self.config.cycle_window, current_cycle)

        # All conditions met - stop!
        return rolling_std < 0.10


class ChurnStoppingCriterion(CycleStoppingCriterion):
    """
    Label-free stopping criterion based on prediction churn over the unlabeled pool.

    The label/search criteria track candidate-BATCH metrics, which are biased toward the
    hardest, selection-picked clips and so make an unreliable convergence signal (they can
    read falsely stable on a weak model, or never settle). This criterion instead measures
    how much the model's PREDICTED labels on the remaining pool change between cycles. When
    the model stops reorganizing the pool, it has converged. This needs no ground truth and
    no candidate metrics - only the per-cycle predictions_v{N}.csv the loop already writes.

    Rule (scale-free, online): stop once the rolling-mean churn has fallen to
    <= churn_peak_frac of its running peak for churn_patience consecutive cycles, after
    cycle_min_cycles. Peak-relative rather than an absolute threshold because the churn
    floor is dataset dependent (pool size, base rate). Churn is a LEADING indicator - it
    quiets slightly before the quality plateau - so it stops a touch conservatively, which
    is the safe direction.
    """

    def __init__(self, config, metrics_history):
        super().__init__(config, metrics_history)
        # version -> {filename: predicted_positive(bool)}; loaded lazily, cached
        self._pred_cache: dict[int, dict[str, bool]] = {}
        # later-version -> flip fraction vs the previous version
        self._churn: dict[int, float] = {}

    def _available_versions(self, max_cycle):
        """Prediction versions on disk, 1..max_cycle (contiguity not assumed)."""
        return [
            v
            for v in range(1, max_cycle + 1)
            if self.config.get_predictions_path(v).exists()
        ]

    def _load_predictions(self, version):
        if version not in self._pred_cache:
            preds = {}
            with self.config.get_predictions_path(version).open(newline="") as f:
                for row in csv.DictReader(f):
                    preds[row["filename"]] = row["prediction"] == "True"
            self._pred_cache[version] = preds
        return self._pred_cache[version]

    def _update_churn(self, versions):
        """Fill in churn for any newly-available consecutive version pairs."""
        for prev_v, cur_v in zip(versions[:-1], versions[1:]):
            if cur_v in self._churn:
                continue
            prev = self._load_predictions(prev_v)
            cur = self._load_predictions(cur_v)
            shared = prev.keys() & cur.keys()
            if shared:
                flips = sum(1 for fn in shared if prev[fn] != cur[fn])
                self._churn[cur_v] = flips / len(shared)

    def _replay(self, versions):
        """Online peak + below-threshold streak over the churn series.

        Returns (fired_cycle | None, rolling_at_last, running_peak, streak_at_last).
        Stateless replay so repeated calls for the same cycle are idempotent.
        """
        churn_versions = [v for v in versions if v in self._churn]
        peak = 0.0
        streak = 0
        fired = None
        roll = None
        window = self.config.cycle_window
        for i, v in enumerate(churn_versions):
            window_vals = [
                self._churn[churn_versions[j]] for j in range(max(0, i - window + 1), i + 1)
            ]
            roll = statistics.mean(window_vals)
            peak = max(peak, roll)  # running peak, updated online
            if v < self.config.cycle_min_cycles or peak == 0.0:
                continue
            streak = streak + 1 if roll <= self.config.churn_peak_frac * peak else 0
            if fired is None and streak >= self.config.churn_patience:
                fired = v
        return fired, roll, peak, streak

    def should_stop(self, current_cycle):
        """Stop when pool-prediction churn has flattened (see class docstring)."""
        if current_cycle < self.config.cycle_min_cycles:
            return False
        versions = self._available_versions(current_cycle)
        if len(versions) < 2:
            return False
        self._update_churn(versions)
        fired = self._replay(versions)[0]
        if fired is not None:
            # Ship the model from the cycle where churn first flattened.
            self.best_cycle = fired
            self.best_rolling_avg = self._churn.get(fired, 0.0)
            return True
        return False

    def status(self, current_cycle):
        """Display snapshot for the workflow: current/rolling/peak churn + streak."""
        versions = self._available_versions(current_cycle)
        if len(versions) >= 2:
            self._update_churn(versions)
        _, roll, peak, streak = self._replay(versions)
        return {
            "current": self._churn.get(current_cycle),
            "rolling": roll,
            "peak": peak,
            "streak": streak,
        }

    def get_status_message(self, current_cycle):
        s = self.status(current_cycle)
        cur = f"{s['current']:.4f}" if s["current"] is not None else "n/a"
        roll = f"{s['rolling']:.4f}" if s["rolling"] is not None else "n/a"
        return (
            f"Pool churn: {cur} (rolling {roll}), peak {s['peak']:.4f}, "
            f"below-threshold streak {s['streak']}/{self.config.churn_patience}"
        )


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
    if strategy == "search":
        return SearchModeStoppingCriterion(config, metrics_history)
    if strategy == "churn":
        return ChurnStoppingCriterion(config, metrics_history)
    if strategy == "none":
        return None
    raise ValueError(
        f"Unknown cycle stopping strategy: '{strategy}'. "
        "Expected: 'label', 'search', 'churn', or 'none'"
    )
