#!/usr/bin/env python3
"""
Tests for auto-labeling utility function.

Tests the ground truth auto-labeling functionality used in evaluation workflows.
"""

import csv

import pytest

from audioloop.utils.auto_labeling import auto_label_from_ground_truth, parse_target_class

_FIELDNAMES_WITH_GT = [
    "filename",
    "predicted_is_positive",
    "confidence",
    "needs_human_label",
    "prediction",
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

# Production mode - no ground truth columns
_FIELDNAMES_NO_GT = [
    "filename",
    "predicted_is_positive",
    "confidence",
    "needs_human_label",
    "prediction",
    "entropy",
    "prob_negative",
    "prob_positive",
    "target_class",
    "original_class",
    "fold",
    "filepath",
]


class TestAutoLabelingUtility:
    """Test cases for the auto-labeling utility function."""

    def test_auto_label_basic_functionality(self, candidates_csv):
        """Test basic auto-labeling with ground truth data."""
        candidates_data = [
            {
                "filename": "test1.wav",
                "ground_truth": "True",
                "predicted_is_positive": "False",
                "confidence": "0.7",
            },
            {
                "filename": "test2.wav",
                "ground_truth": "False",
                "predicted_is_positive": "True",
                "confidence": "0.8",
            },
            {
                "filename": "test3.wav",
                "ground_truth": "True",
                "predicted_is_positive": "True",
                "confidence": "0.9",
            },
        ]

        csv_file = candidates_csv(candidates_data, _FIELDNAMES_WITH_GT)
        results = auto_label_from_ground_truth(str(csv_file))

        # Check return values
        assert results["positive_count"] == 2
        assert results["negative_count"] == 1
        assert results["total"] == 3

        # Verify file was updated in place
        with csv_file.open() as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3
        assert rows[0]["needs_human_label"] == "1"  # True -> 1
        assert rows[1]["needs_human_label"] == "0"  # False -> 0
        assert rows[2]["needs_human_label"] == "1"  # True -> 1

    def test_auto_label_no_ground_truth(self, candidates_csv):
        """Test error handling when no ground truth data is available."""
        candidates_data = [
            {"filename": "test1.wav", "predicted_is_positive": "False", "confidence": "0.7"}
        ]

        # CSV without ground truth columns
        csv_file = candidates_csv(candidates_data, _FIELDNAMES_NO_GT)

        with pytest.raises(ValueError, match="No ground truth data found"):
            auto_label_from_ground_truth(str(csv_file))

    def test_auto_label_file_not_found(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            auto_label_from_ground_truth("/nonexistent/file.csv")

    def test_auto_label_empty_file(self, tmp_path):
        """Test error handling for empty CSV files."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")  # truly empty: no header, no rows

        with pytest.raises(ValueError, match="No candidates found"):
            auto_label_from_ground_truth(str(empty_file))

    def test_auto_label_mixed_boolean_formats(self, candidates_csv):
        """Test handling of different boolean string formats."""
        candidates_data = [
            {"filename": "test1.wav", "ground_truth": "true"},  # lowercase
            {"filename": "test2.wav", "ground_truth": "TRUE"},  # uppercase
            {"filename": "test3.wav", "ground_truth": "True"},  # mixed case
            {"filename": "test4.wav", "ground_truth": "false"},  # lowercase false
            {"filename": "test5.wav", "ground_truth": ""},  # empty string
        ]

        csv_file = candidates_csv(candidates_data, _FIELDNAMES_WITH_GT)
        results = auto_label_from_ground_truth(str(csv_file))

        assert results["positive_count"] == 3  # All 'true' variants
        assert results["negative_count"] == 2  # 'false' and empty

        # Verify labels
        with csv_file.open() as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["needs_human_label"] == "1"  # true
        assert rows[1]["needs_human_label"] == "1"  # TRUE
        assert rows[2]["needs_human_label"] == "1"  # True
        assert rows[3]["needs_human_label"] == "0"  # false
        assert rows[4]["needs_human_label"] == "0"  # empty

    def test_parse_target_class(self):
        """Test target class parsing utility function."""
        # Test positive predictions
        assert parse_target_class("Guitar") == "Guitar"
        assert parse_target_class("Speech") == "Speech"
        assert parse_target_class("Drill") == "Drill"

        # Test negative predictions
        assert parse_target_class("not_Guitar") == "Guitar"
        assert parse_target_class("not_Speech") == "Speech"
        assert parse_target_class("not_Drill") == "Drill"

        # Test edge cases
        assert parse_target_class("not_not_Guitar") == "not_Guitar"  # Double negative
        assert parse_target_class("") == ""  # Empty string


# Integration tests removed - automated_workflow.py is a top-level script
# The utility function tests above provide comprehensive coverage of the core functionality


if __name__ == "__main__":
    pytest.main([__file__])
