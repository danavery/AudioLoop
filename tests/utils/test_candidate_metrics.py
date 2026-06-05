#!/usr/bin/env python3
"""
Tests for candidate metrics persistence utilities.

Tests the saving and loading of candidate metrics across cycles.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audioloop.merge_labels import compute_and_log_candidate_metrics
from audioloop.utils.candidate_metrics import (
    calculate_and_save_candidate_metrics,
    load_candidate_metrics_history,
    save_candidate_metrics,
)

# Column sets for the candidates CSVs consumed by the metrics functions (see the shared
# `candidates_csv` fixture in tests/conftest.py).
_SAVE_FIELDS = [
    "filename",
    "prediction",
    "needs_human_label",
    "target_class",
    "confidence",
    "entropy",
]
_COMPUTE_FIELDS = ["filename", "prediction", "needs_human_label", "target_class", "confidence"]


class TestSaveCandidateMetrics:
    """Test cases for save_candidate_metrics function."""

    def test_save_new_metrics(self):
        """Test saving metrics to a new history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metrics = {"f1_score": 0.75, "precision": 0.80, "recall": 0.70, "accuracy": 0.78}

            save_candidate_metrics(output_dir, cycle=1, metrics=metrics)

            # Verify file was created
            history_file = output_dir / "candidate_metrics_history.json"
            assert history_file.exists()

            # Verify content
            with history_file.open("r") as f:
                data = json.load(f)

            assert "1" in data
            assert data["1"]["cycle"] == 1
            assert data["1"]["f1_score"] == 0.75
            assert data["1"]["precision"] == 0.80

    def test_save_updates_existing_history(self):
        """Test that saving adds to existing history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save first cycle
            metrics1 = {"f1_score": 0.60, "precision": 0.65, "recall": 0.55}
            save_candidate_metrics(output_dir, cycle=1, metrics=metrics1)

            # Save second cycle
            metrics2 = {"f1_score": 0.70, "precision": 0.75, "recall": 0.65}
            save_candidate_metrics(output_dir, cycle=2, metrics=metrics2)

            # Load and verify both are present
            history = load_candidate_metrics_history(output_dir)
            assert len(history) == 2
            assert 1 in history
            assert 2 in history
            assert history[1]["f1_score"] == 0.60
            assert history[2]["f1_score"] == 0.70

    def test_save_overwrites_existing_cycle(self):
        """Test that saving overwrites metrics for an existing cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Save initial metrics
            metrics1 = {"f1_score": 0.60, "precision": 0.65}
            save_candidate_metrics(output_dir, cycle=1, metrics=metrics1)

            # Overwrite same cycle
            metrics2 = {"f1_score": 0.75, "precision": 0.80}
            save_candidate_metrics(output_dir, cycle=1, metrics=metrics2)

            # Verify overwrite
            history = load_candidate_metrics_history(output_dir)
            assert len(history) == 1
            assert history[1]["f1_score"] == 0.75

    def test_atomic_write(self):
        """Test that writes are atomic (temp file is removed on success)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            metrics = {"f1_score": 0.75}

            save_candidate_metrics(output_dir, cycle=1, metrics=metrics)

            # Verify no temp file remains
            temp_file = output_dir / "candidate_metrics_history.json.tmp"
            assert not temp_file.exists()


class TestLoadCandidateMetricsHistory:
    """Test cases for load_candidate_metrics_history function."""

    def test_load_existing_history(self):
        """Test loading an existing history file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            history_file = output_dir / "candidate_metrics_history.json"

            # Create test history
            test_data = {
                "1": {"cycle": 1, "f1_score": 0.60},
                "2": {"cycle": 2, "f1_score": 0.70},
            }
            with history_file.open("w") as f:
                json.dump(test_data, f)

            # Load and verify
            history = load_candidate_metrics_history(output_dir)
            assert len(history) == 2
            assert history[1]["f1_score"] == 0.60
            assert history[2]["f1_score"] == 0.70

    def test_load_nonexistent_file(self):
        """Test loading when history file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            history = load_candidate_metrics_history(output_dir)
            assert history == {}

    def test_load_corrupt_json(self):
        """Test handling of corrupt JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            history_file = output_dir / "candidate_metrics_history.json"

            # Write invalid JSON
            with history_file.open("w") as f:
                f.write("{invalid json")

            # Should return empty dict and not crash
            history = load_candidate_metrics_history(output_dir)
            assert history == {}

    def test_load_converts_string_keys_to_int(self):
        """Test that string keys are converted to integers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            history_file = output_dir / "candidate_metrics_history.json"

            # JSON stores keys as strings
            test_data = {"1": {"f1_score": 0.60}, "10": {"f1_score": 0.70}}
            with history_file.open("w") as f:
                json.dump(test_data, f)

            history = load_candidate_metrics_history(output_dir)
            # Keys should be ints
            assert 1 in history
            assert 10 in history
            assert "1" not in history


class TestCalculateAndSaveCandidateMetrics:
    """Test cases for calculate_and_save_candidate_metrics function."""

    def test_calculate_and_save_success(self, candidates_csv):
        """Test successful calculation and saving of metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test candidates
            candidates_data = [
                {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
                {"filename": "tn1.wav", "prediction": "False", "needs_human_label": "0"},
            ]
            candidates_file = candidates_csv(
                candidates_data, _SAVE_FIELDS, name="labeling_candidates_v1.csv"
            )

            # Calculate and save
            metrics = calculate_and_save_candidate_metrics(candidates_file, output_dir, cycle=1)

            # Verify metrics returned
            assert metrics["num_candidates"] == 2
            assert metrics["accuracy"] == 1.0

            # Verify saved to file
            history = load_candidate_metrics_history(output_dir)
            assert 1 in history
            assert history[1]["accuracy"] == 1.0

    def test_calculate_no_labeled_candidates(self, candidates_csv):
        """Test handling when no candidates are labeled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create unlabeled candidates
            candidates_data = [
                {"filename": "u1.wav", "prediction": "True", "needs_human_label": ""},
            ]
            candidates_file = candidates_csv(
                candidates_data, _SAVE_FIELDS, name="labeling_candidates_v1.csv"
            )

            # Should return empty dict
            metrics = calculate_and_save_candidate_metrics(candidates_file, output_dir, cycle=1)
            assert metrics == {}

            # Should not create history file
            history = load_candidate_metrics_history(output_dir)
            assert history == {}

    def test_calculate_nonexistent_file(self):
        """Test handling when candidates file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            candidates_file = Path(tmpdir) / "nonexistent.csv"

            metrics = calculate_and_save_candidate_metrics(candidates_file, output_dir, cycle=1)
            assert metrics == {}


class TestComputeAndLogCandidateMetrics:
    """Test cases for compute_and_log_candidate_metrics function in merge_labels."""

    def test_compute_with_valid_config(self, candidates_csv):
        """Test computing metrics with valid config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock config
            config = MagicMock()
            config.output_dir = Path(tmpdir)

            # Create test candidates
            candidates_data = [
                {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            ]
            candidates_file = candidates_csv(
                candidates_data, _COMPUTE_FIELDS, name="labeling_candidates_v5.csv"
            )

            # Compute metrics
            metrics = compute_and_log_candidate_metrics(str(candidates_file), config)

            # Should return valid metrics
            assert "f1_score" in metrics
            assert metrics["num_candidates"] == 1

            # Should be saved to history
            history = load_candidate_metrics_history(Path(tmpdir))
            assert 5 in history

    def test_compute_with_no_config(self, candidates_csv):
        """Test that no metrics are computed when config is None."""
        candidates_data = [
            {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
        ]
        candidates_file = candidates_csv(
            candidates_data, _COMPUTE_FIELDS, name="labeling_candidates_v1.csv"
        )

        metrics = compute_and_log_candidate_metrics(str(candidates_file), config=None)
        assert metrics == {}

    def test_compute_with_invalid_filename(self, candidates_csv):
        """Test handling when cycle number can't be extracted from filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.output_dir = Path(tmpdir)

            # Create file with non-standard name
            candidates_data = [
                {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            ]
            candidates_file = candidates_csv(
                candidates_data, _COMPUTE_FIELDS, name="weird_filename.csv"
            )

            metrics = compute_and_log_candidate_metrics(str(candidates_file), config)
            assert metrics == {}

    def test_compute_respects_log_level(self, candidates_csv):
        """Test that log level is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MagicMock()
            config.output_dir = Path(tmpdir)

            candidates_data = [
                {"filename": "tp1.wav", "prediction": "True", "needs_human_label": "1"},
            ]
            candidates_file = candidates_csv(
                candidates_data, _COMPUTE_FIELDS, name="labeling_candidates_v1.csv"
            )

            # Should not raise errors with different log levels
            metrics = compute_and_log_candidate_metrics(
                str(candidates_file), config, log_level=logging.WARNING
            )
            assert "f1_score" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
