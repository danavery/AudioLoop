#!/usr/bin/env python3
"""
Tests for cycle-level stopping criteria.

Tests the stopping logic for active learning loops based on candidate
metrics trends across multiple cycles.
"""

from unittest.mock import MagicMock

import pytest

from audioloop.config import AudioLoopConfig
from audioloop.utils.cycle_stopping_criteria import (
    CycleStoppingCriterion,
    LabelModeStoppingCriterion,
    SearchModeStoppingCriterion,
    create_cycle_stopping_criterion,
)


class TestCycleStoppingCriterionHelpers:
    """Test helper methods in base class."""

    def test_calculate_rolling_avg_full_window(self):
        """Test rolling average with full window of data."""
        config = MagicMock()
        config.cycle_window = 3
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.65},
            3: {"f1_score": 0.70},
        }

        criterion = CycleStoppingCriterion(config, metrics_history)
        avg = criterion._calculate_rolling_avg("f1_score", 3, 3)

        # (0.70 + 0.65 + 0.60) / 3 = 0.65
        assert abs(avg - 0.65) < 0.001

    def test_calculate_rolling_avg_insufficient_data(self):
        """Test rolling average when not enough data for full window."""
        config = MagicMock()
        metrics_history = {
            1: {"f1_score": 0.60},
        }

        criterion = CycleStoppingCriterion(config, metrics_history)
        avg = criterion._calculate_rolling_avg("f1_score", 3, 1)

        # Should return 0.0 when insufficient data
        assert avg == 0.0

    def test_calculate_rolling_avg_with_gaps(self):
        """Test rolling average skips missing cycles."""
        config = MagicMock()
        metrics_history = {
            1: {"f1_score": 0.60},
            3: {"f1_score": 0.70},  # Cycle 2 missing
            4: {"f1_score": 0.75},
        }

        criterion = CycleStoppingCriterion(config, metrics_history)
        avg = criterion._calculate_rolling_avg("f1_score", 3, 4)

        # Should use cycles 4, 3, 1 (skipping missing 2)
        # (0.75 + 0.70 + 0.60) / 3 = 0.683
        assert abs(avg - 0.683) < 0.01

    def test_calculate_rolling_avg_missing_metric_key(self):
        """Test rolling average when metric key is missing from some cycles."""
        config = MagicMock()
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"other_metric": 0.65},  # Missing f1_score
            3: {"f1_score": 0.70},
        }

        criterion = CycleStoppingCriterion(config, metrics_history)
        avg = criterion._calculate_rolling_avg("f1_score", 3, 3)

        # Should skip cycle 2 and return 0.0 (insufficient data)
        assert avg == 0.0

    def test_calculate_rolling_std(self):
        """Test rolling standard deviation calculation."""
        config = MagicMock()
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.65},
            3: {"f1_score": 0.70},
        }

        criterion = CycleStoppingCriterion(config, metrics_history)
        std = criterion._calculate_rolling_std("f1_score", 3, 3)

        # Std of [0.70, 0.65, 0.60] ≈ 0.05
        assert abs(std - 0.05) < 0.01

    def test_calculate_rolling_std_insufficient_data(self):
        """Test rolling std returns inf when insufficient data."""
        config = MagicMock()
        metrics_history = {
            1: {"f1_score": 0.60},
        }

        criterion = CycleStoppingCriterion(config, metrics_history)
        std = criterion._calculate_rolling_std("f1_score", 3, 1)

        assert std == float("inf")

    def test_has_improved(self):
        """Test improvement detection."""
        config = MagicMock()
        config.cycle_min_delta = 0.02
        criterion = CycleStoppingCriterion(config, {})

        # Improved beyond delta (0.721 > 0.70 + 0.02)
        assert criterion._has_improved(0.721, 0.70) is True

        # Exactly at delta threshold (0.72 == 0.70 + 0.02, not greater)
        assert criterion._has_improved(0.72, 0.70) is False

        # Improved but not beyond delta
        assert criterion._has_improved(0.71, 0.70) is False

        # Not improved
        assert criterion._has_improved(0.68, 0.70) is False


class TestLabelModeStoppingCriterion:
    """Test LabelMode stopping logic."""

    def test_stops_when_all_conditions_met(self):
        """Test stopping when patience exhausted and stable."""
        config = AudioLoopConfig(
            cycle_min_cycles=3,
            cycle_patience=2,
            cycle_min_delta=0.02,
            cycle_window=2,
            cycle_std_threshold=0.05,
        )

        # Metrics showing plateau
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.65},
            3: {"f1_score": 0.66},  # Small improvement, starts plateau
            4: {"f1_score": 0.665},  # Within delta of previous rolling avg
            5: {"f1_score": 0.67},  # Still stable, patience should be exhausted
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)

        # Cycle 3: should not stop (just started)
        assert criterion.should_stop(3) is False

        # Cycle 4: should not stop (patience = 1)
        assert criterion.should_stop(4) is False

        # Cycle 5: should stop (patience exhausted, stable)
        assert criterion.should_stop(5) is True

    def test_does_not_stop_before_min_cycles(self):
        """Test doesn't stop before minimum cycles."""
        config = AudioLoopConfig(cycle_min_cycles=10)
        metrics_history = {i: {"f1_score": 0.60} for i in range(1, 6)}

        criterion = LabelModeStoppingCriterion(config, metrics_history)
        assert criterion.should_stop(5) is False

    def test_does_not_stop_if_improving(self):
        """Test doesn't stop while improving."""
        config = AudioLoopConfig(
            cycle_min_cycles=3,
            cycle_patience=2,
            cycle_window=2,
            cycle_min_delta=0.02,
        )

        # Steadily improving
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.65},
            3: {"f1_score": 0.70},
            4: {"f1_score": 0.75},
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)

        # Should not stop while improving
        assert criterion.should_stop(3) is False
        assert criterion.should_stop(4) is False
        assert criterion.cycles_without_improvement == 0

    def test_does_not_stop_if_unstable(self):
        """Test doesn't stop if performance is unstable."""
        config = AudioLoopConfig(
            cycle_min_cycles=3,
            cycle_patience=2,
            cycle_window=3,
            cycle_std_threshold=0.05,
        )

        # High variance
        metrics_history = {
            1: {"f1_score": 0.50},
            2: {"f1_score": 0.70},
            3: {"f1_score": 0.55},
            4: {"f1_score": 0.65},
            5: {"f1_score": 0.60},
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)

        # Even if patience exhausted, should not stop if unstable
        for cycle in range(3, 6):
            if criterion.should_stop(cycle):
                # Check if std is actually below threshold
                std = criterion._calculate_rolling_std("f1_score", 3, cycle)
                assert std < config.cycle_std_threshold

    def test_tracks_best_cycle(self):
        """Test tracks cycle with best rolling average."""
        config = AudioLoopConfig(
            cycle_window=2,
            cycle_min_delta=0.02,
            cycle_min_cycles=2,  # Allow checking to start at cycle 2
        )
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.70},  # Rolling avg = (0.70 + 0.60)/2 = 0.65
            3: {"f1_score": 0.75},  # Rolling avg = (0.75 + 0.70)/2 = 0.725 (best)
            4: {"f1_score": 0.70},  # Rolling avg = (0.70 + 0.75)/2 = 0.725 (same, no improvement)
            5: {"f1_score": 0.65},  # Rolling avg = (0.65 + 0.70)/2 = 0.675 (decline)
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)

        criterion.should_stop(2)
        criterion.should_stop(3)
        best_after_3 = criterion.get_best_cycle()

        criterion.should_stop(4)
        criterion.should_stop(5)
        best_after_5 = criterion.get_best_cycle()

        # Best should be cycle 3 (highest rolling avg)
        assert best_after_3 == 3
        assert best_after_5 == 3  # Should remain 3 even after decline

    def test_insufficient_window_data(self):
        """Test doesn't stop when insufficient data for window."""
        config = AudioLoopConfig(cycle_window=5)
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.65},
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)
        assert criterion.should_stop(2) is False


class TestSearchModeStoppingCriterion:
    """Test SearchMode stopping logic."""

    def test_stops_when_all_conditions_met(self):
        """Test stopping when recall plateaus and precision above floor."""
        config = AudioLoopConfig(
            cycle_min_cycles=3,
            cycle_patience=2,
            cycle_window=2,
            precision_floor="auto",
        )

        metrics_history = {
            1: {"recall": 0.70, "precision": 0.50},  # Floor will be max(0.30, 0.40)=0.40
            2: {"recall": 0.75, "precision": 0.48},
            3: {"recall": 0.76, "precision": 0.47},  # Plateau starts
            4: {"recall": 0.765, "precision": 0.46},
            5: {"recall": 0.77, "precision": 0.45},  # Still above floor
        }

        criterion = SearchModeStoppingCriterion(config, metrics_history)

        # Should eventually stop when stable and above floor
        criterion.should_stop(3)
        criterion.should_stop(4)
        criterion.should_stop(5)

        # Verify precision floor was calculated correctly
        assert criterion._precision_floor == 0.40

    def test_does_not_stop_when_precision_below_floor(self):
        """Test doesn't stop if precision drops below floor."""
        config = AudioLoopConfig(
            cycle_min_cycles=3,
            cycle_patience=2,
            cycle_window=2,
            precision_floor=0.50,  # Manual floor
        )

        # Recall plateaus but precision below floor
        metrics_history = {
            1: {"recall": 0.80, "precision": 0.40},
            2: {"recall": 0.82, "precision": 0.42},
            3: {"recall": 0.83, "precision": 0.43},
            4: {"recall": 0.83, "precision": 0.44},  # Precision still < 0.50
        }

        criterion = SearchModeStoppingCriterion(config, metrics_history)

        for cycle in range(3, 5):
            assert criterion.should_stop(cycle) is False

    def test_manual_precision_floor(self):
        """Test manual precision floor override."""
        config = AudioLoopConfig(precision_floor=0.60)
        metrics_history = {
            1: {"recall": 0.70, "precision": 0.80},
        }

        criterion = SearchModeStoppingCriterion(config, metrics_history)
        assert criterion._precision_floor == 0.60

    def test_auto_precision_floor_calculation(self):
        """Test auto precision floor calculation."""
        config = AudioLoopConfig(precision_floor="auto")

        # Test with initial precision 0.70
        metrics_history = {
            1: {"precision": 0.70, "recall": 0.60},
        }
        criterion = SearchModeStoppingCriterion(config, metrics_history)
        # Floor = max(0.30, 0.70 - 0.1) = 0.60
        assert criterion._precision_floor == 0.60

        # Test with low initial precision
        metrics_history = {
            1: {"precision": 0.20, "recall": 0.90},
        }
        criterion = SearchModeStoppingCriterion(config, metrics_history)
        # Floor = max(0.30, 0.20 - 0.1) = 0.30
        assert criterion._precision_floor == 0.30

    def test_precision_floor_fallback_missing_cycle1(self):
        """Test fallback when cycle 1 metrics not available."""
        config = AudioLoopConfig(precision_floor="auto")
        metrics_history = {
            2: {"precision": 0.70, "recall": 0.60},  # No cycle 1
        }

        criterion = SearchModeStoppingCriterion(config, metrics_history)
        # Should fall back to 0.30
        assert criterion._precision_floor == 0.30

    def test_optimizes_recall_not_f1(self):
        """Test that SearchMode tracks recall, not F1."""
        config = AudioLoopConfig(
            cycle_window=2,
            cycle_min_delta=0.02,
            precision_floor=0.30,
        )

        metrics_history = {
            1: {"recall": 0.60, "f1_score": 0.50, "precision": 0.40},
            2: {"recall": 0.70, "f1_score": 0.45, "precision": 0.38},  # F1 worse, recall better
        }

        criterion = SearchModeStoppingCriterion(config, metrics_history)
        criterion.should_stop(2)

        # Best should be tracked by recall improvement
        assert criterion.cycles_without_improvement == 0  # Recall improved


class TestCycleStoppingFactory:
    """Test factory function."""

    def test_creates_label_mode(self):
        """Test creates LabelMode criterion."""
        config = AudioLoopConfig(cycle_stopping_strategy="label")
        metrics = {1: {"f1_score": 0.60}}

        criterion = create_cycle_stopping_criterion(config, metrics)
        assert isinstance(criterion, LabelModeStoppingCriterion)

    def test_creates_search_mode(self):
        """Test creates SearchMode criterion."""
        config = AudioLoopConfig(cycle_stopping_strategy="search")
        metrics = {1: {"recall": 0.70, "precision": 0.50}}

        criterion = create_cycle_stopping_criterion(config, metrics)
        assert isinstance(criterion, SearchModeStoppingCriterion)

    def test_returns_none_for_none_strategy(self):
        """Test returns None when strategy is 'none'."""
        config = AudioLoopConfig(cycle_stopping_strategy="none")
        metrics = {}

        criterion = create_cycle_stopping_criterion(config, metrics)
        assert criterion is None

    def test_raises_error_for_invalid_strategy(self):
        """Test raises ValueError for invalid strategy."""
        config = AudioLoopConfig(cycle_stopping_strategy="invalid")
        metrics = {}

        with pytest.raises(ValueError, match="Unknown cycle stopping strategy"):
            create_cycle_stopping_criterion(config, metrics)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metrics_history(self):
        """Test handles empty metrics history."""
        config = AudioLoopConfig()
        metrics_history = {}

        criterion = LabelModeStoppingCriterion(config, metrics_history)
        assert criterion.should_stop(1) is False

    def test_single_cycle_history(self):
        """Test handles single cycle history."""
        config = AudioLoopConfig(cycle_window=3)
        metrics_history = {1: {"f1_score": 0.60}}

        criterion = LabelModeStoppingCriterion(config, metrics_history)
        # Should not stop with insufficient window
        assert criterion.should_stop(1) is False

    def test_missing_cycles_in_sequence(self):
        """Test handles gaps in cycle sequence."""
        config = AudioLoopConfig(
            cycle_min_cycles=3,
            cycle_patience=2,
            cycle_window=2,
            cycle_std_threshold=0.05,
        )

        # Cycles 1, 3, 5 (missing 2, 4)
        metrics_history = {
            1: {"f1_score": 0.60},
            3: {"f1_score": 0.65},
            5: {"f1_score": 0.66},
            6: {"f1_score": 0.665},
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)
        # Should handle gracefully, using available cycles
        criterion.should_stop(6)  # Should not crash

    def test_status_message(self):
        """Test status message generation."""
        config = AudioLoopConfig()
        metrics_history = {
            1: {"f1_score": 0.60},
            2: {"f1_score": 0.70},
        }

        criterion = LabelModeStoppingCriterion(config, metrics_history)
        criterion.should_stop(2)

        message = criterion.get_status_message(2)
        assert "Best cycle:" in message
        assert "cycles without improvement:" in message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
