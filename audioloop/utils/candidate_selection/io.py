"""
I/O operations for candidate selection.

This module provides functions for loading predictions and saving candidates
to CSV files for the active learning pipeline.
"""

import csv
import logging
import os
from typing import Any

# Set up logger for this module
logger = logging.getLogger(__name__)


def load_predictions(predictions_file: str) -> list[dict[str, Any]]:
    """
    Load predictions from CSV file.

    Args:
        predictions_file: Path to predictions CSV file

    Returns:
        List of prediction dictionaries with parsed numeric fields
    """
    predictions = []
    with open(predictions_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["entropy"] = float(row["entropy"])
            predictions.append(row)
    return predictions


def save_candidates(candidates: list[dict[str, Any]], candidates_csv: str) -> None:
    """
    Save candidates to CSV file with labeling helper columns.

    Args:
        candidates: List of candidate dictionaries
        candidates_csv: Path to output CSV file
    """
    if not candidates:
        logger.info("No candidates to save.")
        return

    # Add labeling helper columns
    for candidate in candidates:
        candidate["needs_human_label"] = ""  # Empty column for human to fill
        candidate["human_confidence"] = ""  # Human confidence in the label

    # Ensure parent directory exists (path should come from config)
    parent_dir = os.path.dirname(candidates_csv)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Save candidates
    fieldnames = list(candidates[0].keys())
    with open(candidates_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    logger.info(f"Candidates saved to: {candidates_csv}")
