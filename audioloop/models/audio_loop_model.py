"""Base model interface for AudioLoop active learning framework."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class AudioLoopModel(nn.Module, ABC):
    """Abstract base class for all AudioLoop models."""

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch: Dictionary containing model inputs

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def prepare_input(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare input batch for the model's expected format.

        Args:
            batch: Raw batch from dataloader

        Returns:
            Dict containing model-ready inputs
        """
        pass

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

    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        pass
