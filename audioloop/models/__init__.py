"""Model architectures for AudioLoop active learning framework."""

from .audio_loop_model import AudioLoopModel
from .cnn_5layer import CNN5Layer
from .simple_cnn import SimpleCNN

# Model registry - maps model type strings to model classes
MODEL_REGISTRY = {
    "CNN5Layer": CNN5Layer,
    "SimpleCNN": SimpleCNN,
}

# Model creation registry - maps CLI model types to model classes
MODEL_TYPES = {
    "cnn5layer": CNN5Layer,
    "simplecnn": SimpleCNN,
}

__all__ = ["MODEL_REGISTRY", "MODEL_TYPES", "AudioLoopModel", "CNN5Layer", "SimpleCNN"]
