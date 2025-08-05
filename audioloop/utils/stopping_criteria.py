"""
Training stopping criteria for AudioLoop models.

This module provides a pluggable architecture for training stopping decisions,
following the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Any


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
        **kwargs,
    ) -> bool:
        """
        Determine if training should stop.

        Args:
            epoch: Current epoch number (0-based)
            train_accuracy: Training accuracy for current epoch
            train_loss: Training loss for current epoch
            val_accuracy: Validation accuracy (if available)
            val_loss: Validation loss (if available)
            **kwargs: Strategy-specific parameters (e.g., train_dataset for PlateauCriterion)

        Returns:
            True if training should stop, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def update_best_model(self, model_state: Any) -> None:
        """
        Update the best model state if current model is better.

        Args:
            model_state: Current model state dict to potentially save
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_model_state(self) -> Any | None:
        """
        Get the best model state seen so far.

        Returns:
            Best model state dict, or None if no best model tracked
        """
        raise NotImplementedError

    @abstractmethod
    def should_update_best_model(self) -> bool:
        """
        Check if the current model should be saved as the best model.

        Returns:
            True if current model should be saved as best, False otherwise
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state for a new training run."""
        return

    def get_current_mode(self) -> str:
        """Get the current stopping mode for display purposes."""
        return self.__class__.__name__.replace("Criterion", "")


class AccuracyCriterion(TrainingStoppingCriterion):
    """Stop when training accuracy reaches 100% or max epochs is reached."""

    def __init__(self, max_epochs: int = 1000):
        """
        Args:
            max_epochs: Maximum number of epochs to train
        """
        self.max_epochs = max_epochs
        self.best_model_state = None

    def should_stop(
        self,
        epoch: int,
        train_accuracy: float,
        train_loss: float,
        val_accuracy: float | None = None,
        val_loss: float | None = None,
        **kwargs,
    ) -> bool:
        # Mark that we should save this model if we hit 100% accuracy
        if train_accuracy >= 1.0:
            self._should_update_best_model = True
            return True

        # Stop if max epochs reached
        if epoch >= self.max_epochs - 1:
            self._should_update_best_model = False
            return True

        self._should_update_best_model = False
        return False

    def should_update_best_model(self) -> bool:
        """Check if the current model should be saved as the best model."""
        return getattr(self, "_should_update_best_model", False)

    def update_best_model(self, model_state: Any) -> None:
        """Update best model state when accuracy reaches 100%."""
        self.best_model_state = model_state
        self._should_update_best_model = False

    def get_best_model_state(self) -> Any | None:
        """Get the best model state seen so far."""
        return self.best_model_state

    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self.best_model_state = None
        self._should_update_best_model = False


class PlateauCriterion(TrainingStoppingCriterion):
    """Stop when training loss plateaus (stops improving)."""

    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 0.01,
        max_epochs: int = 1000,
        accuracy_floor: float | None = None,
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            max_epochs: Maximum epochs (fallback safety)
            accuracy_floor: Only count patience when accuracy >= this threshold (optional)
                          If None, will be auto-calculated from training data when set_training_context() is called
        """
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.accuracy_floor = accuracy_floor
        self._original_accuracy_floor = accuracy_floor  # Store original value for reset
        self._auto_accuracy_floor = accuracy_floor is None  # Track if we need to auto-calculate
        self.best_train_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model_state = None

    def should_stop(
        self,
        epoch: int,
        train_accuracy: float,
        train_loss: float,
        val_accuracy: float | None = None,
        val_loss: float | None = None,
        **kwargs,
    ) -> bool:
        # Auto-calculate accuracy floor on first call if dataset provided
        if self._auto_accuracy_floor and 'train_dataset' in kwargs:
            positive_ratio = self._calculate_positive_ratio(kwargs['train_dataset'])
            trivial_accuracy = max(positive_ratio, 1 - positive_ratio)
            self.accuracy_floor = min(trivial_accuracy + 0.15, 0.99)
            self._auto_accuracy_floor = False
            print(f"Auto-calculated accuracy floor: {self.accuracy_floor:.2f} (trivial accuracy: {trivial_accuracy:.2f})")

        # Stop immediately if we hit 100% accuracy (best case scenario)
        if train_accuracy >= 1.0:
            # Mark that we should save this model as the best (perfect accuracy)
            self._should_update_best_model = True
            return True

        # Check if training loss improved
        if train_loss < self.best_train_loss - self.min_delta:
            self.best_train_loss = train_loss
            self.epochs_without_improvement = 0
            # Mark that we have a new best model (caller should update model state)
            self._should_update_best_model = True
        else:
            # Only increment patience if accuracy is above floor (if specified)
            if self.accuracy_floor is None or train_accuracy >= self.accuracy_floor:
                self.epochs_without_improvement += 1
                self._should_update_best_model = False
            else:
                # Reset patience if accuracy falls below floor
                self.epochs_without_improvement = 0
                self._should_update_best_model = False

        # Stop if patience exceeded or max epochs reached
        return self.epochs_without_improvement >= self.patience or epoch >= self.max_epochs - 1

    def should_update_best_model(self) -> bool:
        """Check if the current model should be saved as the best model."""
        return getattr(self, "_should_update_best_model", False)

    def update_best_model(self, model_state: Any) -> None:
        """Update best model state when loss improves."""
        self.best_model_state = model_state
        self._should_update_best_model = False

    def get_best_model_state(self) -> Any | None:
        """Get the best model state seen so far."""
        return self.best_model_state

    def _calculate_positive_ratio(self, training_dataset) -> float:
        """Calculate the ratio of positive samples in the training dataset.
        
        Args:
            training_dataset: Training dataset with label information
            
        Returns:
            Ratio of positive samples (label=1) to total samples
        """
        if len(training_dataset) == 0:
            return 0.5  # Default to balanced if no data
        
        positive_count = sum(1 for i in range(len(training_dataset)) if training_dataset[i]["label"] == 1)
        return positive_count / len(training_dataset)

    def reset(self) -> None:
        """Reset internal state for a new training run."""
        self.best_train_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self._should_update_best_model = False
        # Reset to original accuracy floor value
        self.accuracy_floor = self._original_accuracy_floor
        self._auto_accuracy_floor = self._original_accuracy_floor is None


def create_stopping_criterion(config) -> TrainingStoppingCriterion:
    """Create a stopping criterion from configuration.

    Args:
        config: AudioLoopConfig with stopping criterion settings

    Returns:
        TrainingStoppingCriterion instance

    Raises:
        ValueError: If stopping_criterion_type is not recognized
    """
    if config.stopping_criterion_type == "plateau":
        return PlateauCriterion(
            patience=config.patience,
            min_delta=config.min_delta,
            max_epochs=config.max_epochs,
            accuracy_floor=config.accuracy_floor,
        )
    if config.stopping_criterion_type == "accuracy":
        return AccuracyCriterion(max_epochs=config.max_epochs)
    raise ValueError(f"Unknown stopping criterion: {config.stopping_criterion_type}")
