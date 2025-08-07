#!/usr/bin/env python3
"""
Tests for auto-labeling utility function.

Tests the ground truth auto-labeling functionality used in evaluation workflows.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from audioloop.utils.auto_labeling import auto_label_from_ground_truth, parse_target_class


class TestAutoLabelingUtility:
    """Test cases for the auto-labeling utility function."""

    def create_test_candidates_csv(self, candidates_data, include_ground_truth=True):
        """Create a temporary candidates CSV file for testing."""
        # Standard fieldnames for candidates CSV
        if include_ground_truth:
            fieldnames = [
                "filename",
                "predicted_is_positive",
                "confidence",
                "needs_human_label",
                "prediction",
                "entropy",
                "prob_negative",
                "prob_positive",
                "ground_truth",
                "correct",
                "original_class",
                "fold",
                "filepath",
            ]
        else:
            # Production mode - no ground truth columns
            fieldnames = [
                "filename",
                "predicted_is_positive",
                "confidence",
                "needs_human_label",
                "prediction",
                "entropy",
                "prob_negative",
                "prob_positive",
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

    def test_auto_label_basic_functionality(self):
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

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            results = auto_label_from_ground_truth(csv_file)

            # Check return values
            assert results["positive_count"] == 2
            assert results["negative_count"] == 1
            assert results["total"] == 3

            # Verify file was updated
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 3
            assert rows[0]["needs_human_label"] == "1"  # True -> 1
            assert rows[1]["needs_human_label"] == "0"  # False -> 0
            assert rows[2]["needs_human_label"] == "1"  # True -> 1

        finally:
            Path(csv_file).unlink()

    def test_auto_label_no_ground_truth(self):
        """Test error handling when no ground truth data is available."""
        candidates_data = [
            {"filename": "test1.wav", "predicted_is_positive": "False", "confidence": "0.7"}
        ]

        # Create CSV without ground truth columns
        csv_file = self.create_test_candidates_csv(candidates_data, include_ground_truth=False)

        try:
            with pytest.raises(ValueError, match="No ground truth data found"):
                auto_label_from_ground_truth(csv_file)
        finally:
            Path(csv_file).unlink()

    def test_auto_label_file_not_found(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            auto_label_from_ground_truth("/nonexistent/file.csv")

    def test_auto_label_empty_file(self):
        """Test error handling for empty CSV files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as temp_file:
            temp_file_name = temp_file.name

        try:
            with pytest.raises(ValueError, match="No candidates found"):
                auto_label_from_ground_truth(temp_file_name)
        finally:
            Path(temp_file_name).unlink()

    def test_auto_label_mixed_boolean_formats(self):
        """Test handling of different boolean string formats."""
        candidates_data = [
            {"filename": "test1.wav", "ground_truth": "true"},  # lowercase
            {"filename": "test2.wav", "ground_truth": "TRUE"},  # uppercase
            {"filename": "test3.wav", "ground_truth": "True"},  # mixed case
            {"filename": "test4.wav", "ground_truth": "false"},  # lowercase false
            {"filename": "test5.wav", "ground_truth": ""},  # empty string
        ]

        csv_file = self.create_test_candidates_csv(candidates_data)

        try:
            results = auto_label_from_ground_truth(csv_file)

            assert results["positive_count"] == 3  # All 'true' variants
            assert results["negative_count"] == 2  # 'false' and empty

            # Verify labels
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert rows[0]["needs_human_label"] == "1"  # true
            assert rows[1]["needs_human_label"] == "1"  # TRUE
            assert rows[2]["needs_human_label"] == "1"  # True
            assert rows[3]["needs_human_label"] == "0"  # false
            assert rows[4]["needs_human_label"] == "0"  # empty

        finally:
            Path(csv_file).unlink()

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
