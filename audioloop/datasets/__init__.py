"""
Dataset configurations and processors for AudioLoop.

This module provides dataset-specific configurations and processors
for different audio datasets used in the AudioLoop framework.
"""

from .fsd50k import FSD50KConfig, FSD50KProcessor
from .urbansound8k import UrbanSound8KConfig, UrbanSound8KProcessor

__all__ = [
    "FSD50KConfig",
    "FSD50KProcessor",
    "UrbanSound8KConfig",
    "UrbanSound8KProcessor",
]
