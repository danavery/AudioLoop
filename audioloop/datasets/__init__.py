"""
Dataset configurations and processors for AudioLoop.

This module provides dataset-specific configurations and processors
for different audio datasets used in the AudioLoop framework.
"""

from .urbansound8k import UrbanSound8KConfig, UrbanSound8KProcessor
from .fsd50k import FSD50KConfig, FSD50KProcessor

__all__ = [
    "UrbanSound8KConfig",
    "UrbanSound8KProcessor",
    "FSD50KConfig",
    "FSD50KProcessor",
]
