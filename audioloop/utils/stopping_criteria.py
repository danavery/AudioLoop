"""
Training stopping criteria for AudioLoop models.

This module provides a pluggable architecture for training stopping decisions,
following the Strategy pattern.
"""

from abc import ABC, abstractmethod


class TrainingStoppingCriterion(ABC):
    """Base class for training stopping criteria."""

    @abstractmethod
    def should_stop(
        self,
        epoch: int,
        train_accuracy: float,
        train_loss: float,
        val_accuracy: float | None = None,
        val_loss: float | None = None,
    ) -> bool:
        """
        Determine if training should stop.

        Args:
            epoch: Current epoch number (0-based)
            train_accuracy: Training accuracy for current epoch
            train_loss: Training loss for current epoch
            val_accuracy: Validation accuracy (if available)
            val_loss: Validation loss (if available)

        Returns:
            True if training should stop, False otherwise
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state for a new training run."""
        return


class AccuracyCriterion(TrainingStoppingCriterion):
    """Stop when training accuracy reaches 100% or max epochs is reached."""

    def __init__(self, max_epochs: int = 1000):
        """
        Args:
            max_epochs: Maximum number of epochs to train
        """
        self.max_epochs = max_epochs

    def should_stop(
        self,
        epoch: int,
        train_accuracy: float,
        train_loss: float,
        val_accuracy: float | None = None,
        val_loss: float | None = None,
    ) -> bool:
        # Stop if we've reached 100% accuracy or max epochs
        return train_accuracy >= 1.0 or epoch >= self.max_epochs - 1
