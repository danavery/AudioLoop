"""Base model interface for AudioLoop active learning framework."""

from abc import ABC, abstractmethod
import torch


class AudioLoopModel(ABC):
    """Abstract base class for all AudioLoop models."""

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to disk with metadata."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, device: torch.device) -> "AudioLoopModel":
        """Load a model from disk."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model metadata."""
        pass
