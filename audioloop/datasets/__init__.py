"""
Dataset configurations for AudioLoop.

This module provides dataset-specific configurations
for different audio datasets used in the AudioLoop framework.
"""

from .fsd50k_config import FSD50KConfig
from .urbansound8k_config import UrbanSound8KConfig

__all__ = [
    "FSD50KConfig",
    "UrbanSound8KConfig",
]
