"""Base model interface for AudioLoop active learning framework."""

from abc import ABC, abstractmethod

import torch.nn as nn


class AudioLoopModel(nn.Module, ABC):
    """Abstract base class for all AudioLoop models.

    Follows standard PyTorch conventions with minimal additional requirements.
    Models should implement forward() normally and provide metadata via get_model_info().
    """

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model metadata for tracking and logging.

        Returns:
            Dict with at minimum: model_type, num_classes, num_parameters
        """
        pass

    @abstractmethod
    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        """Check if this model can handle tensors with the given shape.

        Args:
            shape: Tensor shape (excluding batch dimension)

        Returns:
            True if the model can process tensors of this shape
        """
        pass
