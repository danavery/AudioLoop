"""
Stratified uncertainty sampling strategy.

Guarantees balanced selection from both predicted classes using uncertainty sampling
within each predicted class. Designed for extremely imbalanced datasets where
standard uncertainty sampling often results in all-negative selections.
"""

import random
from typing import Any

from .base import CandidateSelectionStrategy


class StratifiedUncertaintyStrategy(CandidateSelectionStrategy):
    """
    Stratified uncertainty sampling that guarantees selection from both predicted classes.
    
    Uses uncertainty sampling within each predicted class to handle extreme imbalance
    where uncertainty sampling across all predictions often results in all-negative selections.
    """
    
    def __init__(self, min_positives: int = 5, target_positive_ratio: float = 0.4):
        """
        Initialize stratified uncertainty strategy.
        
        Args:
            min_positives: Minimum positive candidates to try to select (default: 5)
            target_positive_ratio: Target ratio of positives when enough available (default: 0.4)
        """
        self.min_positives = min_positives
        self.target_positive_ratio = target_positive_ratio
    
    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        positive_class_name: str = "positive",
        negative_class_name: str = "negative",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Select candidates using stratified uncertainty sampling."""
        random_seed = kwargs.get("random_seed", 42)
        random.seed(random_seed)
        
        # Separate predictions by predicted class
        positive_preds = [p for p in predictions if p["prediction"] == positive_class_name]
        negative_preds = [p for p in predictions if p["prediction"] == negative_class_name]
        
        # Sort each group by entropy (highest first = most uncertain)
        positive_preds.sort(key=lambda x: x["entropy"], reverse=True)
        negative_preds.sort(key=lambda x: x["entropy"], reverse=True)
        
        # Determine how many positives to select
        available_positives = len(positive_preds)
        
        if available_positives == 0:
            # No predicted positives - fall back to pure entropy sampling
            print("  No predicted positives available, using pure entropy sampling")
            all_preds = predictions.copy()
            all_preds.sort(key=lambda x: x["entropy"], reverse=True)
            selected = all_preds[:num_candidates]
            random.shuffle(selected)
            return selected
        
        # Calculate target number of positives
        target_positives_by_ratio = int(num_candidates * self.target_positive_ratio)
        target_positives = min(
            available_positives,  # Can't take more than available
            max(self.min_positives, target_positives_by_ratio)  # At least min_positives
        )
        
        # Don't take more positives than requested candidates
        target_positives = min(target_positives, num_candidates)
        
        # Select candidates
        selected_positives = positive_preds[:target_positives]
        
        # Fill remaining slots with negatives
        remaining_slots = num_candidates - len(selected_positives)
        selected_negatives = negative_preds[:remaining_slots]
        
        # Combine and shuffle
        all_candidates = selected_positives + selected_negatives
        random.shuffle(all_candidates)
        
        # Print selection summary
        print(f"  Stratified selection: {len(selected_positives)} positive + {len(selected_negatives)} negative")
        print(f"  Available predictions: {available_positives} positive, {len(negative_preds)} negative")
        
        return all_candidates
    
    def get_name(self) -> str:
        return "Stratified Uncertainty Sampling"