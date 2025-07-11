"""Model architectures for AudioLoop active learning framework."""

from .base import AudioLoopModel
from .cnn_5layer import SoundCNN
from .simple_cnn import SimpleCNN

# Model registry - maps model type strings to model classes
MODEL_REGISTRY = {
    "SoundCNN": SoundCNN,
    "SimpleCNN": SimpleCNN,
}

# Model creation registry - maps CLI model types to model classes
MODEL_TYPES = {
    "soundcnn": SoundCNN,
    "simplecnn": SimpleCNN,
}

__all__ = ["MODEL_REGISTRY", "MODEL_TYPES", "AudioLoopModel", "SimpleCNN", "SoundCNN"]
