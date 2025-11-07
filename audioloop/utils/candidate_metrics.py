"""
Candidate metrics persistence utilities.

This module provides functions for saving and loading candidate metrics
across active learning cycles, enabling stopping criteria to track performance trends.
"""

import json
import logging
from pathlib import Path

from .metrics_utils import calculate_candidate_metrics

logger = logging.getLogger(__name__)


def save_candidate_metrics(output_dir: Path, cycle: int, metrics: dict[str, float]) -> None:
    """
    Save candidate metrics for a cycle to the metrics history file.

    Args:
        output_dir: Experiment output directory
        cycle: Cycle number
        metrics: Metrics dict from calculate_candidate_metrics()

    The metrics are saved to {output_dir}/candidate_metrics_history.json.
    Uses atomic writes (temp file + rename) to prevent corruption.
    """
    history_file = output_dir / "candidate_metrics_history.json"

    # Load existing history
    history = load_candidate_metrics_history(output_dir)

    # Add cycle number to metrics for convenience
    metrics_with_cycle = {"cycle": cycle, **metrics}

    # Update history with new metrics
    history[cycle] = metrics_with_cycle

    # Write atomically using temp file
    temp_file = history_file.with_suffix(".json.tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(
                {str(k): v for k, v in history.items()}, f, indent=2, sort_keys=True
            )
        # Atomic rename (POSIX guarantees atomicity)
        temp_file.replace(history_file)
    except OSError as e:
        logger.warning(f"Failed to save candidate metrics: {e}")
        if temp_file.exists():
            temp_file.unlink()


def load_candidate_metrics_history(output_dir: Path) -> dict[int, dict[str, float]]:
    """
    Load candidate metrics history from JSON file.

    Args:
        output_dir: Experiment output directory

    Returns:
        Dict mapping cycle number (int) to metrics dict.
        Returns empty dict if file doesn't exist or is corrupt.
    """
    history_file = output_dir / "candidate_metrics_history.json"

    if not history_file.exists():
        return {}

    try:
        with history_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert string keys to int
        return {int(k): v for k, v in data.items()}

    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning(f"Failed to load candidate metrics history: {e}")
        return {}


def calculate_and_save_candidate_metrics(
    candidates_file: Path, output_dir: Path, cycle: int
) -> dict[str, float]:
    """
    Calculate candidate metrics and save to history file.

    This is a convenience function that combines calculate_candidate_metrics()
    and save_candidate_metrics().

    Args:
        candidates_file: Path to candidates CSV file
        output_dir: Experiment output directory
        cycle: Cycle number

    Returns:
        Metrics dict if successful, empty dict otherwise
    """
    # Calculate metrics
    metrics = calculate_candidate_metrics(candidates_file)

    if not metrics:
        logger.warning(
            f"No candidate metrics calculated for cycle {cycle} "
            f"(file: {candidates_file.name})"
        )
        return {}

    # Save to history
    save_candidate_metrics(output_dir, cycle, metrics)

    logger.info(
        f"Saved candidate metrics for cycle {cycle}: "
        f"{metrics['num_candidates']} candidates"
    )

    return metrics
