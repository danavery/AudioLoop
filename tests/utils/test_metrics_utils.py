#!/usr/bin/env python3
"""
Tests for metrics calculation utilities.

Tests the candidate metrics calculation functionality used in stopping criteria.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from audioloop.utils.metrics_utils import calculate_candidate_metrics


class TestCalculateCandidateMetrics:
    """Test cases for the calculate_candidate_metrics function."""

    def create_test_candidates_csv(self, candidates_data):
        """Create a temporary candidates CSV file for testing."""
        fieldnames = [
            "filename",
            "prediction",
            "confidence",
            "needs_human_label",
            "entropy",
            "prob_negative",
            "prob_positive",
            "ground_truth",
            "correct",
            "original_class",
            "fold",
            "filepath",
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()

            for data in candidates_data:
                # Fill in defaults for missing fields
                row = dict.fromkeys(fieldnames, "")
                row.update(data)
                writer.writerow(row)

            return temp_file.name

    def test_basic_functionality(self):
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

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

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

        finally:
            Path(csv_file).unlink()

    def test_perfect_performance(self):
        """Test metrics when model predictions perfectly match human labels."""
        candidates_data = [
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tp2.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            {"filename": "tn2.wav", "prediction": "False", "needs_human_label": "0"},
        ]

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0
            assert metrics["f1_score"] == 1.0
            assert metrics["accuracy"] == 1.0
            assert metrics["false_positives"] == 0
            assert metrics["false_negatives"] == 0

        finally:
            Path(csv_file).unlink()

    def test_all_predictions_wrong(self):
        """Test metrics when all predictions are incorrect."""
        candidates_data = [
            # Model predicts positive, human says negative
            {"filename": "fp1.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp2.wav", "prediction": "True", "needs_human_label": "0"},
            # Model predicts negative, human says positive
            {"filename": "fn1.wav", "prediction": "False", "needs_human_label": "1"},
            {"filename": "fn2.wav", "prediction": "False", "needs_human_label": "1"},
        ]

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            assert metrics["precision"] == 0.0
            assert metrics["recall"] == 0.0
            assert metrics["f1_score"] == 0.0
            assert metrics["accuracy"] == 0.0
            assert metrics["true_positives"] == 0
            assert metrics["true_negatives"] == 0

        finally:
            Path(csv_file).unlink()

    def test_high_recall_low_precision(self):
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

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            # Recall should be 1.0 (no false negatives)
            assert metrics["recall"] == 1.0
            # Precision should be low (3 / 10 = 0.3)
            assert abs(metrics["precision"] - 0.3) < 0.001

        finally:
            Path(csv_file).unlink()

    def test_unlabeled_candidates_ignored(self):
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

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            # Should only count the 2 labeled candidates
            assert metrics["num_candidates"] == 2
            assert metrics["true_positives"] == 1
            assert metrics["true_negatives"] == 1
            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0

        finally:
            Path(csv_file).unlink()

    def test_boolean_string_prediction_formats(self):
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

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            # All should be correctly classified
            assert metrics["accuracy"] == 1.0
            assert metrics["true_positives"] == 3
            assert metrics["true_negatives"] == 3

        finally:
            Path(csv_file).unlink()

    def test_nonexistent_file(self):
        """Test handling of non-existent file."""
        metrics = calculate_candidate_metrics(Path("/nonexistent/file.csv"))
        assert metrics == {}

    def test_empty_candidates_file(self):
        """Test handling of CSV file with no candidate rows."""
        candidates_data = []
        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))
            assert metrics == {}
        finally:
            Path(csv_file).unlink()

    def test_no_labeled_candidates(self):
        """Test handling of file where all candidates are unlabeled."""
        candidates_data = [
            {"filename": "unlabeled1.wav", "prediction": "True", "needs_human_label": ""},
            {"filename": "unlabeled2.wav", "prediction": "False", "needs_human_label": ""},
            {"filename": "unlabeled3.wav", "prediction": "True", "needs_human_label": ""},
        ]

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))
            assert metrics == {}
        finally:
            Path(csv_file).unlink()

    def test_all_positive_predictions(self):
        """Test edge case where model predicts all positive."""
        candidates_data = [
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            {"filename": "fp1.wav", "prediction": "True", "needs_human_label": "0"},
            {"filename": "fp2.wav", "prediction": "True", "needs_human_label": "0"},
        ]

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            # Precision = 1/3, Recall = 1/1 = 1.0
            assert abs(metrics["precision"] - 1 / 3) < 0.001
            assert metrics["recall"] == 1.0
            assert metrics["false_negatives"] == 0
            assert metrics["true_negatives"] == 0

        finally:
            Path(csv_file).unlink()

    def test_all_negative_predictions(self):
        """Test edge case where model predicts all negative."""
        candidates_data = [
            {"filename": "fn1.wav", "prediction": "False", "needs_human_label": "1"},
            {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            {"filename": "tn2.wav", "prediction": "False", "needs_human_label": "0"},
        ]

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            # Precision = 0/0 = 0.0, Recall = 0/1 = 0.0
            assert metrics["precision"] == 0.0
            assert metrics["recall"] == 0.0
            assert metrics["f1_score"] == 0.0
            assert metrics["true_positives"] == 0
            assert metrics["false_positives"] == 0

        finally:
            Path(csv_file).unlink()

    def test_ground_truth_not_used(self):
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

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            metrics = calculate_candidate_metrics(Path(csv_file))

            # Should use needs_human_label, not ground_truth
            # TP: match.wav (pred=True, human=1)
            # FN: differ.wav (pred=False, human=1)
            assert metrics["true_positives"] == 1
            assert metrics["false_negatives"] == 1
            assert metrics["false_positives"] == 0
            assert metrics["true_negatives"] == 0

        finally:
            Path(csv_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
