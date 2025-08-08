"""
Confidence-based candidate selection strategy.

Selects candidates with highest model confidence scores, using random sampling
within a pool of high-confidence predictions to improve diversity.
"""

import random
from typing import Any

from .base import CandidateSelectionStrategy


class ConfidenceStrategy(CandidateSelectionStrategy):
    """Select candidates with highest confidence scores."""

    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select high-confidence candidates using existing algorithm."""
        positive_percentage = kwargs.get("positive_percentage", 0.8)
        candidate_pool_multiplier = kwargs.get("candidate_pool_multiplier", 5)
        random_seed = kwargs.get("random_seed", 42)

        # Calculate numbers based on percentage
        num_positive = int(num_candidates * positive_percentage)
        num_negative = num_candidates - num_positive

        # Separate positive and negative predictions
        positive_preds = [p for p in predictions if p["predicted_class"] == positive_class_name]
        negative_preds = [p for p in predictions if p["predicted_class"] == negative_class_name]

        # Sort by confidence (highest first)
        positive_preds.sort(key=lambda x: x["confidence"], reverse=True)
        negative_preds.sort(key=lambda x: x["confidence"], reverse=True)

        # Create broader candidate pools for sampling
        positive_pool_size = min(len(positive_preds), num_positive * candidate_pool_multiplier)
        negative_pool_size = min(len(negative_preds), num_negative * candidate_pool_multiplier)

        positive_pool = positive_preds[:positive_pool_size]
        negative_pool = negative_preds[:negative_pool_size]

        # Randomly sample from the pools to improve diversity
        random.seed(random_seed)

        positive_candidates = random.sample(positive_pool, min(num_positive, len(positive_pool)))
        negative_candidates = random.sample(negative_pool, min(num_negative, len(negative_pool)))

        # Combine and shuffle final candidates
        all_candidates = positive_candidates + negative_candidates
        random.shuffle(all_candidates)

        return all_candidates

    def get_name(self) -> str:
        return "Confidence-Based"
