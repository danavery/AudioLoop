"""
Random candidate selection strategy (baseline).

Selects candidates completely at random, regardless of model predictions.
This serves as a baseline to compare against active learning strategies.
"""

import random
from typing import Any

from .base import CandidateSelectionStrategy


class RandomStrategy(CandidateSelectionStrategy):
    """Select candidates randomly (baseline strategy)."""

    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select candidates randomly.

        This is a pure random baseline that ignores all model outputs
        (confidence, entropy, predicted class). It provides a baseline
        to measure whether intelligent selection strategies actually
        provide value over random selection.

        Args:
            predictions: List of prediction dictionaries
            num_candidates: Total number of candidates to select
            **kwargs: Accepts random_seed for reproducibility

        Returns:
            List of randomly selected candidates
        """
        random_seed = kwargs.get("random_seed", 42)

        # Randomly sample from all predictions
        random.seed(random_seed)
        candidates = random.sample(predictions, min(num_candidates, len(predictions)))
        random.shuffle(candidates)

        return candidates

    def get_name(self) -> str:
        return "Random (Baseline)"
