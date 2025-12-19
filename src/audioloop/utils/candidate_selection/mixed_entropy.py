"""
Mixed-entropy candidate selection strategy.

Samples from multiple entropy levels to balance learning value (hard examples)
with representative evaluation (varied difficulty). Selects from high, medium,
and low entropy buckets using fixed distribution ratios.
"""

import random
from typing import Any

from .base import CandidateSelectionStrategy


class MixedEntropyStrategy(CandidateSelectionStrategy):
    """Select candidates across entropy levels (70% high / 20% medium / 10% low)."""

    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select candidates from mixed entropy levels.

        Samples 70% from high-entropy (top 20%), 20% from medium-entropy (next 40%),
        and 10% from low-entropy (bottom 40%) predictions.

        Args:
            predictions: List of prediction dictionaries with 'entropy' key
            num_candidates: Total number of candidates to select
            positive_class_name: Name of positive class (unused in mixed mode)
            negative_class_name: Name of negative class (unused in mixed mode)
            **kwargs: Additional parameters (candidate_pool_multiplier, random_seed, positive_percentage)

        Returns:
            List of selected candidate dictionaries

        Raises:
            ValueError: If positive_percentage is set (incompatible with mixed mode)
        """
        # Validate mutual exclusivity with stratification
        positive_percentage = kwargs.get("positive_percentage")
        if positive_percentage is not None:
            raise ValueError(
                "mixed_entropy selection mode is incompatible with positive_percentage stratification. "
                "Set positive_percentage=None to use mixed entropy sampling."
            )

        candidate_pool_multiplier = kwargs.get("candidate_pool_multiplier", 5)
        random_seed = kwargs.get("random_seed", 42)

        # Set random seed for reproducibility
        random.seed(random_seed)

        # Sort all predictions by entropy (highest first = most uncertain)
        sorted_preds = sorted(predictions, key=lambda x: x["entropy"], reverse=True)
        n = len(sorted_preds)

        # Handle edge case: very small datasets
        if n < 100:
            # Fall back to pure high-entropy sampling for small datasets
            # Use a smaller effective multiplier to maintain high-entropy focus
            effective_multiplier = min(candidate_pool_multiplier, 1.5)
            pool_size = min(n, int(num_candidates * effective_multiplier))
            pool = sorted_preds[:pool_size]
            candidates = random.sample(pool, min(num_candidates, len(pool)))
            random.shuffle(candidates)
            return candidates

        # Define percentile buckets (20/40/40 split)
        top_20_idx = int(n * 0.20)
        top_60_idx = int(n * 0.60)

        high_entropy_pool = sorted_preds[0:top_20_idx]          # Top 20%
        medium_entropy_pool = sorted_preds[top_20_idx:top_60_idx]  # Next 40%
        low_entropy_pool = sorted_preds[top_60_idx:]            # Bottom 40%

        # Calculate target samples for each bucket (70/20/10 distribution)
        high_target = int(num_candidates * 0.70)
        medium_target = int(num_candidates * 0.20)
        low_target = num_candidates - high_target - medium_target  # Remainder to handle rounding

        # Sample from each bucket with pool multiplier for diversity
        high_samples = self._sample_from_pool(
            high_entropy_pool, high_target, candidate_pool_multiplier, random_seed
        )
        medium_samples = self._sample_from_pool(
            medium_entropy_pool, medium_target, candidate_pool_multiplier, random_seed
        )
        low_samples = self._sample_from_pool(
            low_entropy_pool, low_target, candidate_pool_multiplier, random_seed
        )

        # Handle shortfalls by redistributing to other buckets
        all_candidates = high_samples + medium_samples + low_samples
        shortfall = num_candidates - len(all_candidates)

        if shortfall > 0:
            # Try to get more from pools that weren't fully utilized
            remaining_pools = []
            used_files = {c["filename"] for c in all_candidates}

            # Collect remaining candidates from each pool (excluding already selected)
            for pool in [high_entropy_pool, medium_entropy_pool, low_entropy_pool]:
                remaining = [p for p in pool if p["filename"] not in used_files]
                remaining_pools.extend(remaining)

            # Sample from remaining candidates
            if remaining_pools:
                additional = random.sample(
                    remaining_pools,
                    min(shortfall, len(remaining_pools))
                )
                all_candidates.extend(additional)

        # Shuffle to mix entropy levels
        random.shuffle(all_candidates)

        return all_candidates

    def _sample_from_pool(
        self,
        pool: list[dict[str, Any]],
        target: int,
        pool_multiplier: int,
        random_seed: int,
    ) -> list[dict[str, Any]]:
        """Sample from a pool with diversity via pool multiplier.

        Args:
            pool: Source pool to sample from
            target: Target number of samples
            pool_multiplier: Multiplier for pool size (for diversity)
            random_seed: Random seed (used for consistency)

        Returns:
            List of sampled candidates
        """
        if not pool:
            return []

        # Create broader pool for diversity
        pool_size = min(len(pool), target * pool_multiplier)
        sampling_pool = pool[:pool_size]

        # Sample target number (or as many as available)
        num_to_sample = min(target, len(sampling_pool))
        samples = random.sample(sampling_pool, num_to_sample)

        return samples

    def get_name(self) -> str:
        return "Mixed Entropy (Representative + Boundary)"
