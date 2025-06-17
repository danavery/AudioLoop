"""Model architectures for AudioLoop active learning framework."""

from .cnn_5layer import SoundCNN
from .simple_cnn import SimpleCNN

__all__ = ["SimpleCNN", "SoundCNN"]
