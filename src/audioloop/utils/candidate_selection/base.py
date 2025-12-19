"""
Base class for candidate selection strategies.

This module provides the abstract base class for all candidate selection
strategies in AudioLoop's active learning pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any


class CandidateSelectionStrategy(ABC):
    """Base class for candidate selection strategies."""

    @abstractmethod
    def select_candidates(
        self,
        predictions: list[dict[str, Any]],
        num_candidates: int,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Select candidates from predictions for human labeling.

        Args:
            predictions: List of prediction dictionaries
            num_candidates: Total number of candidates to select
            **kwargs: Strategy-specific parameters

        Returns:
            List of selected candidates
        """
        raise NotImplementedError

    def get_name(self) -> str:
        """Return strategy name for logging."""
        return self.__class__.__name__

    def get_active_strategy_name(self) -> str:
        """Return active strategy name for user-facing output. Override for dynamic strategies."""
        return self.get_name()
