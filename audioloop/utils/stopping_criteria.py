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


class PlateauCriterion(TrainingStoppingCriterion):
    """Stop when training loss plateaus (stops improving)."""

    def __init__(self, patience: int = 20, min_delta: float = 0.01, max_epochs: int = 1000):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            max_epochs: Maximum epochs (fallback safety)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.best_train_loss = float("inf")
        self.epochs_without_improvement = 0

    def should_stop(
        self,
        epoch: int,
        train_accuracy: float,
        train_loss: float,
        val_accuracy: float | None = None,
        val_loss: float | None = None,
    ) -> bool:
        # Stop immediately if we hit 100% accuracy (best case scenario)
        if train_accuracy >= 1.0:
            return True

        # Check if training loss improved
        if train_loss < self.best_train_loss - self.min_delta:
            self.best_train_loss = train_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Stop if patience exceeded or max epochs reached
        return self.epochs_without_improvement >= self.patience or epoch >= self.max_epochs - 1

    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self.best_train_loss = float("inf")
        self.epochs_without_improvement = 0
