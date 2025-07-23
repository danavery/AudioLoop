"""
Dataset configurations for AudioLoop.

This module provides dataset-specific configurations
for different audio datasets used in the AudioLoop framework.
"""

from .fsd50k import FSD50KConfig
from .urbansound8k import UrbanSound8KConfig

__all__ = [
    "FSD50KConfig",
    "UrbanSound8KConfig",
]
