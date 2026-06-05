#!/usr/bin/env python3
"""
Tests for metrics calculation utilities.

Tests the candidate metrics calculation functionality used in stopping criteria.
"""

from pathlib import Path

import pytest

from audioloop.utils.metrics_utils import calculate_candidate_metrics

_FIELDNAMES = [
    "filename",
    "prediction",
    "confidence",
    "needs_human_label",
    "entropy",
    "prob_negative",
    "prob_positive",
    "target_class",
    "ground_truth",
    "correct",
    "original_class",
    "fold",
    "filepath",
]


class TestCalculateCandidateMetrics:
    """Test cases for the calculate_candidate_metrics function."""

    def test_basic_functionality(self, candidates_csv):
        """Test basic metrics calculation with labeled candidates."""
        candidates_data = [
            # True Positives (2)
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tp2.wav", "prediction": "True", "needs_human_label": "1"},
            # False Positives (1)
            {"filename": "fp1.wav", "prediction": "True", "needs_human_label": "0"},
            # False Negatives (1)
            {"filename": "fn1.wav", "prediction": "False", "needs_human_label": "1"},
            # True Negatives (2)
            {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            {"filename": "tn2.wav", "prediction": "False", "needs_human_label": "0"},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # Check confusion matrix
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_negatives"] == 2
        assert metrics["num_candidates"] == 6

        # Check calculated metrics
        # Precision = TP / (TP + FP) = 2 / 3 = 0.667
        assert abs(metrics["precision"] - 2 / 3) < 0.001
        # Recall = TP / (TP + FN) = 2 / 3 = 0.667
        assert abs(metrics["recall"] - 2 / 3) < 0.001
        # F1 = 2 * (P * R) / (P + R) = 2 * (2/3 * 2/3) / (4/3) = 0.667
        assert abs(metrics["f1_score"] - 2 / 3) < 0.001
        # Accuracy = (TP + TN) / total = 4 / 6 = 0.667
        assert abs(metrics["accuracy"] - 4 / 6) < 0.001

    def test_perfect_performance(self, candidates_csv):
        """Test metrics when model predictions perfectly match human labels."""
        candidates_data = [
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tp2.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            {"filename": "tn2.wav", "prediction": "False", "needs_human_label": "0"},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["false_positives"] == 0
        assert metrics["false_negatives"] == 0

    def test_all_predictions_wrong(self, candidates_csv):
        """Test metrics when all predictions are incorrect."""
        candidates_data = [
            # Model predicts positive, human says negative
            {"filename": "fp1.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp2.wav", "prediction": "True", "needs_human_label": "0"},
            # Model predicts negative, human says positive
            {"filename": "fn1.wav", "prediction": "False", "needs_human_label": "1"},
            {"filename": "fn2.wav", "prediction": "False", "needs_human_label": "1"},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["true_positives"] == 0
        assert metrics["true_negatives"] == 0

    def test_high_recall_low_precision(self, candidates_csv):
        """Test metrics for high recall but low precision scenario."""
        candidates_data = [
            # True Positives (3)
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tp2.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tp3.wav", "prediction": "True", "needs_human_label": "1"},
            # False Positives (7) - many false alarms
            {"filename": "fp1.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp2.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp3.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp4.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp5.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp6.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp7.wav", "prediction": "True", "needs_human_label": "0"},
            # No False Negatives - caught all positives
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # Recall should be 1.0 (no false negatives)
        assert metrics["recall"] == 1.0
        # Precision should be low (3 / 10 = 0.3)
        assert abs(metrics["precision"] - 0.3) < 0.001

    def test_unlabeled_candidates_ignored(self, candidates_csv):
        """Test that unlabeled candidates are ignored in metrics calculation."""
        candidates_data = [
            # Labeled candidates
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            # Unlabeled candidates (should be ignored)
            {"filename": "unlabeled1.wav", "prediction": "True", "needs_human_label": ""},
            {"filename": "unlabeled2.wav", "prediction": "False", "needs_human_label": ""},
            {"filename": "unlabeled3.wav", "prediction": "True", "needs_human_label": "   "},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # Should only count the 2 labeled candidates
        assert metrics["num_candidates"] == 2
        assert metrics["true_positives"] == 1
        assert metrics["true_negatives"] == 1
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_boolean_string_prediction_formats(self, candidates_csv):
        """Test handling of different prediction string formats."""
        candidates_data = [
            # Different True formats
            {"filename": "t1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "t2.wav", "prediction": "true", "needs_human_label": "1"},
            {"filename": "t3.wav", "prediction": "TRUE", "needs_human_label": "1"},
            # Different False formats
            {"filename": "f1.wav", "prediction": "False", "needs_human_label": "0"},
            {"filename": "f2.wav", "prediction": "false", "needs_human_label": "0"},
            {"filename": "f3.wav", "prediction": "FALSE", "needs_human_label": "0"},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # All should be correctly classified
        assert metrics["accuracy"] == 1.0
        assert metrics["true_positives"] == 3
        assert metrics["true_negatives"] == 3

    def test_nonexistent_file(self):
        """Test handling of non-existent file."""
        metrics = calculate_candidate_metrics(Path("/nonexistent/file.csv"))
        assert metrics == {}

    def test_empty_candidates_file(self, candidates_csv):
        """Test handling of CSV file with no candidate rows."""
        metrics = calculate_candidate_metrics(candidates_csv([], _FIELDNAMES))
        assert metrics == {}

    def test_no_labeled_candidates(self, candidates_csv):
        """Test handling of file where all candidates are unlabeled."""
        candidates_data = [
            {"filename": "unlabeled1.wav", "prediction": "True", "needs_human_label": ""},
            {"filename": "unlabeled2.wav", "prediction": "False", "needs_human_label": ""},
            {"filename": "unlabeled3.wav", "prediction": "True", "needs_human_label": ""},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))
        assert metrics == {}

    def test_all_positive_predictions(self, candidates_csv):
        """Test edge case where model predicts all positive."""
        candidates_data = [
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "fp1.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp2.wav", "prediction": "True", "needs_human_label": "0"},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # Precision = 1/3, Recall = 1/1 = 1.0
        assert abs(metrics["precision"] - 1 / 3) < 0.001
        assert metrics["recall"] == 1.0
        assert metrics["false_negatives"] == 0
        assert metrics["true_negatives"] == 0

    def test_all_negative_predictions(self, candidates_csv):
        """Test edge case where model predicts all negative."""
        candidates_data = [
            {"filename": "fn1.wav", "prediction": "False", "needs_human_label": "1"},
            {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            {"filename": "tn2.wav", "prediction": "False", "needs_human_label": "0"},
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # Precision = 0/0 = 0.0, Recall = 0/1 = 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["true_positives"] == 0
        assert metrics["false_positives"] == 0

    def test_ground_truth_not_used(self, candidates_csv):
        """Test that ground_truth column is never used, only needs_human_label."""
        candidates_data = [
            # Ground truth and human label match
            {
                "filename": "match.wav",
                "prediction": "True",
                "needs_human_label": "1",
                "ground_truth": "True",
            },
            # Ground truth and human label DIFFER (human label should be used)
            {
                "filename": "differ.wav",
                "prediction": "False",
                "needs_human_label": "1",
                "ground_truth": "False",
            },
        ]

        metrics = calculate_candidate_metrics(candidates_csv(candidates_data, _FIELDNAMES))

        # Should use needs_human_label, not ground_truth
        # TP: match.wav (pred=True, human=1)
        # FN: differ.wav (pred=False, human=1)
        assert metrics["true_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["false_positives"] == 0
        assert metrics["true_negatives"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
