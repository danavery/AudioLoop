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

# Import factory function
from .factory import create_strategy

# Import I/O functions
from .io import load_predictions, save_candidates
from .mixed_entropy import MixedEntropyStrategy
from .random import RandomStrategy

# Import statistics/reporting functions
from .statistics import print_selection_statistics
from .stratified import StratifiedUncertaintyStrategy

# Public API
__all__ = [
    # Base class
    "CandidateSelectionStrategy",
    # Strategy implementations
    "BasicTransitionStrategy",
    "ConfidenceStrategy",
    "EntropyStrategy",
    "MixedEntropyStrategy",
    "RandomStrategy",
    "StratifiedUncertaintyStrategy",
    # Factory function
    "create_strategy",
    # I/O functions
    "load_predictions",
    "save_candidates",
    # Statistics/reporting functions
    "print_selection_statistics",
]
