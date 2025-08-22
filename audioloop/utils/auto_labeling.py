#!/usr/bin/env python3
"""
Auto-labeling utility for candidates with ground truth data.

This module provides functionality to automatically label candidates using
ground truth data from the 'ground_truth' column in candidates CSV files.

This is intended for evaluation/research workflows where ground truth is available,
not for production workflows with unlabeled data.
"""

import csv
import logging
from pathlib import Path

# Set up logger for this module
logger = logging.getLogger(__name__)


def auto_label_from_ground_truth(candidates_csv: str, log_level=logging.INFO) -> dict[str, int]:
    """
    Automatically label candidates using ground truth data.

    Reads candidates CSV file, uses the 'ground_truth' column to set
    'needs_human_label' field, and saves the results back to the same file.

    Args:
        candidates_csv: Path to candidates CSV with 'ground_truth' column
        log_level: Logging level (logging.INFO for normal output, logging.WARNING for quiet)

    Returns:
        dict: Statistics with keys 'positive_count', 'negative_count', 'total'

    Raises:
        FileNotFoundError: If candidates CSV file doesn't exist
        ValueError: If no ground truth data available or invalid format
    """
    # Set up logging for this auto-labeling run
    logger.setLevel(log_level)

    candidates_path = Path(candidates_csv)

    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_csv}")

    # Load candidates
    candidates = []
    fieldnames = []

    try:
        with open(candidates_path) as f:
            reader = csv.DictReader(f)
            candidates = list(reader)
            fieldnames = reader.fieldnames
    except Exception as e:
        raise ValueError(f"Error reading candidates CSV: {e}") from e

    if not candidates:
        raise ValueError("No candidates found in CSV file")

    if not fieldnames:
        raise ValueError("No fieldnames found in CSV file")

    # Validate ground truth column exists
    has_ground_truth = any("ground_truth" in candidate for candidate in candidates)
    if not has_ground_truth:
        raise ValueError(
            "No ground truth data found. The 'ground_truth' column is required "
            "for auto-labeling. This suggests the CSV was generated in production mode "
            "rather than evaluation mode."
        )

    positive_count = 0
    negative_count = 0

    # Apply ground truth labels
    logger.info(f"Auto-labeling {len(candidates)} candidates using ground truth...")

    for i, candidate in enumerate(candidates):
        ground_truth = candidate.get("ground_truth", "")

        # Convert ground truth to label
        if str(ground_truth).lower() == "true":
            candidate["needs_human_label"] = "1"
            positive_count += 1
        else:
            candidate["needs_human_label"] = "0"
            negative_count += 1

        # Show progress for large files
        if (i + 1) % 100 == 0 or i == len(candidates) - 1:
            logger.info(f"Processed {i + 1}/{len(candidates)} samples...")

    # Save results back to file
    try:
        with open(candidates_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)
    except Exception as e:
        raise ValueError(f"Error saving labeled candidates: {e}") from e

    results = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "total": len(candidates),
    }

    logger.info("Auto-labeling complete:")
    logger.info(f"  Positive samples: {positive_count}")
    logger.info(f"  Negative samples: {negative_count}")
    logger.info(f"  Total samples: {len(candidates)}")
    logger.info(f"  Results saved to: {candidates_csv}")

    return results


def parse_target_class(prediction: str) -> str:
    """
    Parse the target class from predicted_class field.

    Helper function for extracting class names from predicted_class strings
    that may have "not_" prefixes for negative predictions.

    Args:
        prediction: Predicted class string like "Guitar" or "not_Guitar"

    Returns:
        Target class name (e.g., "Guitar")
    """
    if prediction.startswith("not_"):
        return prediction[4:]  # Remove "not_" prefix
    return prediction
