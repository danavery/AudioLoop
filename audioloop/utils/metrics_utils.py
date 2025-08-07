"""
Metrics calculation utilities for AudioLoop.

This module provides functions for calculating performance metrics
from predictions, supporting both active learning and evaluation workflows.
"""

import statistics
from typing import Any

import numpy as np


def calculate_binary_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculate binary classification metrics from predictions.

    Args:
        predictions: List of prediction dictionaries with keys:
            - predicted_is_positive: bool or str (required)
            - confidence: float (required)
            - ground_truth: bool or str (optional, required for evaluation metrics)

    Returns:
        Dict containing calculated metrics:
            - mean_confidence, std_confidence, etc.: Always available
            - f1_score, precision, recall, accuracy: Only when ground truth available
            - predicted_positive_ratio: Always available
    """
    if not predictions:
        return {}

    # Check if ground truth is available
    has_ground_truth = all(
        "ground_truth" in p and p["ground_truth"] is not None for p in predictions if p
    )

    # Always calculate prediction and confidence metrics
    pred_labels = []
    confidences = []

    for p in predictions:
        pred_pos = p["predicted_is_positive"]
        if isinstance(pred_pos, str):
            pred_pos = pred_pos.lower() == "true"
        pred_labels.append(int(pred_pos))
        confidences.append(float(p["confidence"]))

    # Calculate confidence statistics (always available)
    mean_confidence = statistics.mean(confidences)
    std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
    median_confidence = statistics.median(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)

    # Calculate prediction distribution (always available)
    predicted_positives = sum(pred_labels)
    predicted_positive_ratio = predicted_positives / len(predictions)

    # Base metrics (always available)
    result = {
        "mean_confidence": mean_confidence,
        "std_confidence": std_confidence,
        "median_confidence": median_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "predicted_positive_ratio": predicted_positive_ratio,
        "predicted_positives": predicted_positives,
        "total_samples": len(predictions),
        "has_ground_truth": has_ground_truth,
    }

    # Add evaluation metrics only if ground truth is available
    if has_ground_truth:
        true_labels = []
        for p in predictions:
            ground_truth_value = p["ground_truth"]
            if isinstance(ground_truth_value, str):
                ground_truth_value = ground_truth_value.lower() == "true"
            true_labels.append(int(ground_truth_value))

        # Calculate confusion matrix components
        tp = sum(1 for t, p in zip(true_labels, pred_labels, strict=False) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_labels, pred_labels, strict=False) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(true_labels, pred_labels, strict=False) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(true_labels, pred_labels, strict=False) if t == 0 and p == 0)

        # Calculate evaluation metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        negative_class_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        actual_positives = sum(true_labels)
        actual_positive_ratio = actual_positives / len(predictions)

        # Add evaluation metrics to result
        result.update(
            {
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "negative_class_accuracy": negative_class_accuracy,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "actual_positive_ratio": actual_positive_ratio,
                "actual_positives": actual_positives,
            }
        )

    return result


def calculate_entropy_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculate entropy-based metrics from predictions.

    Args:
        predictions: List of prediction dictionaries with 'entropy' key

    Returns:
        Dict containing entropy metrics:
            - mean_entropy: Mean entropy across predictions
            - std_entropy: Standard deviation of entropy
            - median_entropy: Median entropy
            - min_entropy: Minimum entropy
            - max_entropy: Maximum entropy
    """
    if not predictions:
        return {}

    entropies = []
    for p in predictions:
        if "entropy" in p and p["entropy"] is not None:
            entropies.append(float(p["entropy"]))

    if not entropies:
        return {}

    return {
        "mean_entropy": statistics.mean(entropies),
        "std_entropy": statistics.stdev(entropies) if len(entropies) > 1 else 0.0,
        "median_entropy": statistics.median(entropies),
        "min_entropy": min(entropies),
        "max_entropy": max(entropies),
    }


def calculate_class_balance_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculate class balance metrics from predictions.

    Args:
        predictions: List of prediction dictionaries

    Returns:
        Dict containing class balance metrics:
            - predicted_positive_ratio: Ratio of predicted positive samples (always available)
            - actual_positive_ratio: Ratio of actual positive samples (only with ground truth)
            - balance_difference: Absolute difference between actual and predicted ratios (only with ground truth)
    """
    if not predictions:
        return {}

    # Check if ground truth is available
    has_ground_truth = all(
        "ground_truth" in p and p["ground_truth"] is not None for p in predictions if p
    )

    predicted_positives = 0
    total = len(predictions)

    # Always calculate predicted positive ratio
    for p in predictions:
        pred_pos = p["predicted_is_positive"]
        if isinstance(pred_pos, str):
            pred_pos = pred_pos.lower() == "true"
        if pred_pos:
            predicted_positives += 1

    predicted_ratio = predicted_positives / total
    result = {"predicted_positive_ratio": predicted_ratio}

    # Add ground truth metrics if available
    if has_ground_truth:
        true_positives = 0
        for p in predictions:
            ground_truth_value = p["ground_truth"]
            if isinstance(ground_truth_value, str):
                ground_truth_value = ground_truth_value.lower() == "true"
            if ground_truth_value:
                true_positives += 1

        actual_ratio = true_positives / total
        result.update(
            {
                "actual_positive_ratio": actual_ratio,
                "balance_difference": abs(actual_ratio - predicted_ratio),
            }
        )

    return result


def calculate_confidence_percentiles(
    predictions: list[dict[str, Any]], percentiles: list[float] | None = None
) -> dict[str, float]:
    """
    Calculate confidence percentiles from predictions.

    Args:
        predictions: List of prediction dictionaries with 'confidence' key
        percentiles: List of percentiles to calculate (default: [5, 95])

    Returns:
        Dict with percentile values (e.g., {"p05_confidence": 0.85, "p95_confidence": 0.98})
    """
    if not predictions:
        return {}

    if percentiles is None:
        percentiles = [5, 95]

    confidences = [float(p["confidence"]) for p in predictions]

    if not confidences:
        return {}

    result = {}

    for p in percentiles:
        # Format percentile as p05, p95, etc.
        key = f"p{p:02.0f}_confidence"
        result[key] = np.percentile(confidences, p)

    return result
