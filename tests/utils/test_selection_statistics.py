"""Tests for print_selection_statistics: end-of-cycle reporting.

This is pure reporting, so the contract worth testing is "never crashes on the data
shapes it actually receives" — it runs after all the expensive inference in a cycle, so
an unhandled TypeError here would kill the run at the finish line. Inputs arrive in two
shapes: native bools (in-memory results) and strings like "True"/"False" (CSV
round-trips), which is exactly what convert_to_bool exists to absorb.
"""

import logging

import pytest

from audioloop.utils.candidate_selection.statistics import print_selection_statistics

LOGGER_NAME = "audioloop.utils.candidate_selection.statistics"


def _pred(name, predicted_class, confidence, **extra):
    return {
        "filename": name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        **extra,
    }


def _predictions(with_ground_truth=False, as_strings=False):
    """Four predictions, 3 of 4 correct when ground truth is attached."""

    def gt(ground_truth, correct):
        if not with_ground_truth:
            return {}
        if as_strings:  # the CSV round-trip shape
            return {"ground_truth": str(ground_truth), "correct": str(correct)}
        return {"ground_truth": ground_truth, "correct": correct}

    return [
        _pred("a.pt", "positive", 0.95, **gt(True, True)),
        _pred("b.pt", "positive", 0.60, **gt(False, False)),
        _pred("c.pt", "negative", 0.85, **gt(False, True)),
        _pred("d.pt", "negative", 0.90, **gt(True, True)),
    ]


@pytest.fixture
def info_caplog(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return caplog


def test_empty_predictions_is_a_noop(info_caplog):
    print_selection_statistics([], [], strategy_name="Entropy")

    assert "No predictions available." in info_caplog.text


def test_without_ground_truth_reports_confidence_only(info_caplog):
    preds = _predictions(with_ground_truth=False)

    print_selection_statistics(preds, preds[:2], strategy_name="Entropy")

    assert "Ground truth evaluation not available" in info_caplog.text
    assert "Overall Confidence" in info_caplog.text
    assert "Selection strategy: Entropy" in info_caplog.text


@pytest.mark.parametrize("as_strings", [False, True], ids=["bool-values", "csv-strings"])
def test_ground_truth_accuracy_identical_for_bool_and_string_shapes(info_caplog, as_strings):
    """The accuracy math must not depend on whether flags survived a CSV round-trip:
    3 of 4 correct -> 0.750 either way."""
    preds = _predictions(with_ground_truth=True, as_strings=as_strings)

    print_selection_statistics(preds, preds[:2], strategy_name="Entropy")

    assert "Overall Accuracy: 0.750" in info_caplog.text
    # Per-class: both true positives correct (2/2), one of two true negatives (1/2).
    assert "True positive Accuracy: 1.000 (2/2)" in info_caplog.text
    assert "True negative Accuracy: 0.500 (1/2)" in info_caplog.text


def test_empty_selection_with_predictions_does_not_crash(info_caplog):
    """A strategy can legitimately select nothing; reporting must survive that."""
    preds = _predictions(with_ground_truth=True)

    print_selection_statistics(preds, [], strategy_name="Entropy")

    assert "Total candidates for labeling: 0" in info_caplog.text


def test_high_accuracy_confident_model_triggers_recommendation(info_caplog):
    """An accurate+confident model under a confidence strategy logs the switch hint."""
    preds = [
        _pred(f"p{i}.pt", "positive", 0.97, ground_truth=True, correct=True) for i in range(20)
    ]

    print_selection_statistics(preds, preds[:3], strategy_name="Confidence-based")

    assert "Consider switching to entropy-based selection" in info_caplog.text


def test_candidate_summaries_report_confidence_ranges(info_caplog):
    preds = _predictions(with_ground_truth=False)

    print_selection_statistics(preds, preds, strategy_name="Entropy")

    assert "positive confidence range: 0.600 - 0.950" in info_caplog.text
    assert "negative confidence range: 0.850 - 0.900" in info_caplog.text
