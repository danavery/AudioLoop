"""
Candidate selection strategies for active learning.

This package provides a pluggable architecture for candidate selection,
following the same pattern as the stopping_criteria module.
"""

# Import base class
from .base import CandidateSelectionStrategy

# Import all strategy implementations
from .basic_transition import BasicTransitionStrategy
from .confidence import ConfidenceStrategy
from .entropy import EntropyStrategy
from .random import RandomStrategy
from .stratified import StratifiedUncertaintyStrategy

# Import utility functions
from .utils import (
    create_strategy,
    load_predictions,
    print_selection_statistics,
    save_candidates,
)

# Public API
__all__ = [
    "BasicTransitionStrategy",
    # Base class
    "CandidateSelectionStrategy",
    # Strategy implementations
    "ConfidenceStrategy",
    "EntropyStrategy",
    "RandomStrategy",
    "StratifiedUncertaintyStrategy",
    # Utility functions
    "create_strategy",
    "load_predictions",
    "print_selection_statistics",
    "save_candidates",
]
