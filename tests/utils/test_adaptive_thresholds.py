"""Tests for AdaptiveThresholdCalculator: the auto-threshold source for BasicTransition.

The existing basic-transition tests exercise the *consumer* with this calculator mocked
out, so until now the real arithmetic never ran under test. These pin the safety clamps
(the invariant that matters most), the banded adjustments, and one unmocked integration
through BasicTransitionStrategy(auto_thresholds=True).
"""

import pytest

from audioloop.utils.adaptive_thresholds import AdaptiveThresholdCalculator, threshold_calculator
from audioloop.utils.candidate_selection.basic_transition import BasicTransitionStrategy


@pytest.fixture
def calc():
    return AdaptiveThresholdCalculator()


class TestSafetyBounds:
    @pytest.mark.parametrize("positive_pct", [0.001, 0.01, 0.03, 0.07, 0.15, 0.5, 1.0])
    @pytest.mark.parametrize("training_set_size", [None, 10, 75, 300, 1000])
    def test_thresholds_always_within_clamps(self, calc, positive_pct, training_set_size):
        """Whatever the inputs, results stay inside the documented safety bounds — the
        guarantee BasicTransition relies on to never get an unusable threshold."""
        f1, confidence, variance = calc.calculate_thresholds(positive_pct, training_set_size)

        assert 0.25 <= f1 <= 0.5
        assert 0.65 <= confidence <= 0.92
        assert 0.08 <= variance <= 0.3


class TestImbalanceBands:
    def test_rarer_class_lowers_confidence_threshold(self, calc):
        """Rarer positives -> less reliable confidence -> more aggressive (lower) threshold."""
        _, conf_very_rare, _ = calc.calculate_thresholds(0.01)
        _, conf_rare, _ = calc.calculate_thresholds(0.03)
        _, conf_imbalanced, _ = calc.calculate_thresholds(0.07)
        _, conf_balanced, _ = calc.calculate_thresholds(0.15)

        assert conf_very_rare < conf_rare < conf_imbalanced < conf_balanced

    def test_f1_threshold_bands(self, calc):
        """F1 = base * max(0.7, sqrt(pct/0.1)), clamped to [0.25, 0.5]."""
        f1_baseline, _, _ = calc.calculate_thresholds(0.1)
        assert f1_baseline == pytest.approx(0.35)  # factor exactly 1.0 at the 10% baseline

        f1_rare, _, _ = calc.calculate_thresholds(0.025)
        assert f1_rare == pytest.approx(0.25)  # 0.35 * 0.7 = 0.245, clamped up to the floor

        f1_common, _, _ = calc.calculate_thresholds(0.4)
        assert f1_common == pytest.approx(0.5)  # 0.35 * sqrt(4.0) = 0.7, clamped down to the cap

    def test_variance_threshold_bands(self, calc):
        _, _, var_rare = calc.calculate_thresholds(0.03)
        _, _, var_common = calc.calculate_thresholds(0.15)

        assert var_rare == pytest.approx(0.26)  # base 0.18 + 0.08 for rare classes
        assert var_common == pytest.approx(0.21)  # base 0.18 + 0.03 otherwise


class TestTrainingSetSizeBands:
    def test_smaller_sets_get_lower_confidence_threshold(self, calc):
        """At a fixed 15% prevalence (no clamping in play): tiny sets transition far more
        aggressively than large ones."""
        _, conf_tiny, _ = calc.calculate_thresholds(0.15, training_set_size=10)
        _, conf_small, _ = calc.calculate_thresholds(0.15, training_set_size=75)
        _, conf_mid, _ = calc.calculate_thresholds(0.15, training_set_size=300)
        _, conf_large, _ = calc.calculate_thresholds(0.15, training_set_size=1000)

        # base 0.85 - 0.02 (balanced) = 0.83, then -0.12 / -0.08 / +0 / +0.01.
        assert conf_tiny == pytest.approx(0.71)
        assert conf_small == pytest.approx(0.75)
        assert conf_mid == pytest.approx(0.83)
        assert conf_large == pytest.approx(0.84)


class TestPrintAnalysis:
    @pytest.mark.parametrize(
        ("positive_pct", "training_set_size"), [(0.01, 10), (0.03, 75), (0.07, None), (0.15, 1000)]
    )
    def test_runs_for_every_band_without_crashing(self, calc, capsys, positive_pct, training_set_size):
        f1, confidence, variance = calc.calculate_thresholds(positive_pct, training_set_size)

        calc.print_analysis(positive_pct, f1, confidence, variance, training_set_size)

        out = capsys.readouterr().out
        assert "Adaptive Threshold Analysis" in out
        assert f"{f1:.3f}" in out


class TestBasicTransitionIntegration:
    def test_auto_thresholds_use_real_calculator(self, capsys):
        """Unmocked integration: BasicTransition(auto_thresholds=True) gets the same values
        the module-level calculator produces, inside the safety bounds."""
        strategy = BasicTransitionStrategy(auto_thresholds=True, estimated_positive_pct=0.05)
        capsys.readouterr()  # swallow the ctor's print_analysis output

        expected_f1, expected_confidence, expected_variance = (
            threshold_calculator.calculate_thresholds(0.05, None)
        )
        assert (
            strategy.f1_threshold,
            strategy.confidence_threshold,
            strategy.variance_threshold,
        ) == (expected_f1, expected_confidence, expected_variance)
        assert 0.25 <= expected_f1 <= 0.5
        assert 0.65 <= expected_confidence <= 0.92
        assert 0.08 <= expected_variance <= 0.3
