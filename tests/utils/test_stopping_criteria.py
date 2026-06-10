"""
Tests for training stopping criteria.

This module tests the criterion objects in isolation (stop decisions, patience/min_delta
accounting, best-model bookkeeping). Integration with the real training loop — that
execute_training_loop honors should_stop/should_update_best_model — lives in
tests/test_training_core.py::TestExecuteTrainingLoop, which drives the production loop
with scripted epoch results.
"""

import pytest

from audioloop.utils.stopping_criteria import (
    AccuracyCriterion,
    PlateauCriterion,
    TrainingStoppingCriterion,
)

# Test constants
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_PATIENCE = 50
DEFAULT_MIN_DELTA = 0.01
PERFECT_ACCURACY = 1.0

# Common test values
TYPICAL_ACCURACY = 0.8
TYPICAL_LOSS = 0.3
HIGH_ACCURACY = 0.95
LOW_ACCURACY = 0.5
HIGH_LOSS = 0.8
LOW_LOSS = 0.1


@pytest.fixture
def training_sequence():
    """Realistic training sequence with plateauing loss."""
    return [
        (0, 0.5, 0.8),  # Initial
        (1, 0.6, 0.7),  # Good improvement
        (2, 0.65, 0.68),  # Small improvement (less than min_delta)
        (3, 0.7, 0.69),  # No improvement (worse)
        (4, 0.72, 0.685),  # Tiny improvement (less than min_delta)
        (5, 0.75, 0.69),  # No improvement
    ]


def assert_stops_at_perfect_accuracy(criterion):
    """Helper to verify criterion stops at 100% accuracy."""
    test_cases = [
        (0, PERFECT_ACCURACY, 0.1),
        (50, PERFECT_ACCURACY, 0.05),
        (999, PERFECT_ACCURACY, 0.0),
    ]

    for epoch, accuracy, loss in test_cases:
        assert criterion.should_stop(epoch=epoch, train_accuracy=accuracy, train_loss=loss)


def assert_ignores_validation_metrics(criterion, epoch=50):
    """Helper to verify criterion ignores validation metrics."""
    result_without_val = criterion.should_stop(
        epoch=epoch, train_accuracy=TYPICAL_ACCURACY, train_loss=TYPICAL_LOSS
    )
    result_with_val = criterion.should_stop(
        epoch=epoch,
        train_accuracy=TYPICAL_ACCURACY,
        train_loss=TYPICAL_LOSS,
        val_accuracy=0.6,
        val_loss=0.5,
    )
    assert result_without_val == result_with_val


class TestTrainingStoppingCriterion:
    """Test the abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that TrainingStoppingCriterion cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TrainingStoppingCriterion()  # type: ignore[abstract]

    def test_reset_has_default_implementation(self):
        """Test that reset() has a default no-op implementation."""

        class ConcreteTestCriterion(TrainingStoppingCriterion):
            def should_stop(
                self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None, **kwargs
            ):
                return False

            def should_update_best_model(self):
                return False

            def update_best_model(self, model_state):
                pass

            def get_best_model_state(self):
                return None

        criterion = ConcreteTestCriterion()
        criterion.reset()  # Should not raise


class TestAccuracyCriterionBestModelSaving:
    """Test best model saving behavior for AccuracyCriterion."""

    def test_saves_best_model_at_perfect_accuracy(self):
        """Test that AccuracyCriterion saves model when hitting 100% accuracy."""
        criterion = AccuracyCriterion(max_epochs=100)
        criterion.reset()

        # Should not update best model for regular accuracy
        should_stop = criterion.should_stop(0, 0.8, 0.5)
        assert not should_stop
        assert not criterion.should_update_best_model()

        # Should update best model when hitting 100% accuracy
        should_stop = criterion.should_stop(1, 1.0, 0.1)
        assert should_stop
        assert criterion.should_update_best_model()

        # Simulate saving the model
        model_state = {"dummy": "state"}
        criterion.update_best_model(model_state)

        # Should be able to retrieve the best model
        best_state = criterion.get_best_model_state()
        assert best_state == model_state

    def test_no_best_model_without_perfect_accuracy(self):
        """Test that no best model is saved if perfect accuracy is never reached."""
        criterion = AccuracyCriterion(max_epochs=3)
        criterion.reset()

        # Run through epochs without hitting 100% accuracy
        for epoch in range(3):
            should_stop = criterion.should_stop(epoch, 0.8, 0.3)
            if not should_stop:
                assert not criterion.should_update_best_model()

        # Should have no best model state
        best_state = criterion.get_best_model_state()
        assert best_state is None

    def test_best_model_state_preserved_after_reset(self):
        """Test that reset() clears the best model state."""
        criterion = AccuracyCriterion(max_epochs=100)
        criterion.reset()

        # Save a model at perfect accuracy
        criterion.should_stop(0, 1.0, 0.1)
        criterion.update_best_model({"saved": "state"})
        assert criterion.get_best_model_state() is not None

        # Reset should clear the best model
        criterion.reset()
        assert criterion.get_best_model_state() is None

    def test_should_update_best_model_resets_after_update(self):
        """Test that should_update_best_model flag is reset after update."""
        criterion = AccuracyCriterion(max_epochs=100)
        criterion.reset()

        # Hit perfect accuracy
        criterion.should_stop(0, 1.0, 0.1)
        assert criterion.should_update_best_model()

        # Update the model
        criterion.update_best_model({"state": "data"})

        # Flag should be reset after update
        assert not criterion.should_update_best_model()


class TestAccuracyCriterion:
    """Test the AccuracyCriterion stopping strategy."""

    def test_has_correct_defaults(self):
        """Test that AccuracyCriterion has correct default parameters."""
        criterion = AccuracyCriterion()
        assert criterion.max_epochs == DEFAULT_MAX_EPOCHS

    def test_accepts_custom_max_epochs(self):
        """Test that AccuracyCriterion accepts custom max_epochs."""
        custom_epochs = 500
        criterion = AccuracyCriterion(max_epochs=custom_epochs)
        assert criterion.max_epochs == custom_epochs

    def test_stops_at_perfect_accuracy(self):
        """Test that AccuracyCriterion stops when training accuracy reaches 100%."""
        criterion = AccuracyCriterion(max_epochs=100)
        assert_stops_at_perfect_accuracy(criterion)

    def test_stops_at_max_epochs(self):
        """Test that AccuracyCriterion stops when max_epochs is reached."""
        criterion = AccuracyCriterion(max_epochs=100)
        # Should stop at max_epochs - 1 (0-based indexing)
        assert criterion.should_stop(
            epoch=99, train_accuracy=HIGH_ACCURACY, train_loss=TYPICAL_LOSS
        )
        assert criterion.should_stop(
            epoch=100, train_accuracy=HIGH_ACCURACY, train_loss=TYPICAL_LOSS
        )

    @pytest.mark.parametrize(
        ("epoch", "accuracy", "loss"),
        [
            (0, LOW_ACCURACY, HIGH_LOSS),
            (50, TYPICAL_ACCURACY, TYPICAL_LOSS),
            (98, 0.99, LOW_LOSS),
        ],
    )
    def test_does_not_stop_early(self, epoch, accuracy, loss):
        criterion = AccuracyCriterion(max_epochs=100)

        assert not criterion.should_stop(epoch=epoch, train_accuracy=accuracy, train_loss=loss)

    def test_ignores_validation_metrics(self):
        """Test that AccuracyCriterion ignores validation metrics."""
        criterion = AccuracyCriterion(max_epochs=100)
        assert_ignores_validation_metrics(criterion)

    def test_reset_is_no_op(self):
        """Test that AccuracyCriterion reset() is a no-op."""
        criterion = AccuracyCriterion(max_epochs=100)
        result_before = criterion.should_stop(
            epoch=50, train_accuracy=TYPICAL_ACCURACY, train_loss=TYPICAL_LOSS
        )
        criterion.reset()
        result_after = criterion.should_stop(
            epoch=50, train_accuracy=TYPICAL_ACCURACY, train_loss=TYPICAL_LOSS
        )
        assert result_before == result_after

    def test_stops_immediately_when_max_epochs_is_one(self):
        criterion = AccuracyCriterion(max_epochs=1)

        assert criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)

    def test_does_not_stop_just_below_perfect_accuracy(self):
        """99.999% is not 100%: the perfect-accuracy stop must be an exact comparison."""
        criterion = AccuracyCriterion(max_epochs=100)

        assert not criterion.should_stop(epoch=50, train_accuracy=0.99999, train_loss=LOW_LOSS)


class TestPlateauCriterion:
    """Test the PlateauCriterion stopping strategy."""

    def test_has_correct_defaults(self):
        """Test that PlateauCriterion has reasonable default parameters."""
        criterion = PlateauCriterion()
        assert criterion.patience == DEFAULT_PATIENCE
        assert criterion.min_delta == DEFAULT_MIN_DELTA
        assert criterion.max_epochs == DEFAULT_MAX_EPOCHS
        assert criterion.best_train_loss == float("inf")
        assert criterion.epochs_without_improvement == 0

    def test_accepts_custom_parameters(self):
        """Test that PlateauCriterion accepts custom parameters."""
        custom_patience = 20
        custom_min_delta = 0.001
        custom_max_epochs = 500

        criterion = PlateauCriterion(
            patience=custom_patience, min_delta=custom_min_delta, max_epochs=custom_max_epochs
        )

        assert criterion.patience == custom_patience
        assert criterion.min_delta == custom_min_delta
        assert criterion.max_epochs == custom_max_epochs

    def test_stops_at_perfect_accuracy(self):
        """Test that PlateauCriterion stops immediately at 100% accuracy."""
        criterion = PlateauCriterion(patience=3, min_delta=0.01, max_epochs=100)
        assert_stops_at_perfect_accuracy(criterion)

    def test_stops_at_max_epochs(self):
        """Test that PlateauCriterion stops when max_epochs is reached."""
        criterion = PlateauCriterion(patience=10, max_epochs=100)

        # Should stop at max_epochs - 1 (0-based indexing)
        assert criterion.should_stop(
            epoch=99, train_accuracy=TYPICAL_ACCURACY, train_loss=TYPICAL_LOSS
        )
        assert criterion.should_stop(
            epoch=100, train_accuracy=TYPICAL_ACCURACY, train_loss=TYPICAL_LOSS
        )

    def test_tracks_best_loss(self):
        """Test that PlateauCriterion tracks the best training loss."""
        criterion = PlateauCriterion(patience=3, min_delta=0.01, max_epochs=100)
        # First epoch should set best loss
        assert not criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)
        assert criterion.best_train_loss == HIGH_LOSS
        assert criterion.epochs_without_improvement == 0

        # Improvement should reset counter
        improved_loss = 0.7
        assert not criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=improved_loss)
        assert criterion.best_train_loss == improved_loss
        assert criterion.epochs_without_improvement == 0

    def test_counts_epochs_without_improvement(self):
        """Test that PlateauCriterion counts epochs without improvement."""
        criterion = PlateauCriterion(patience=3, min_delta=0.01, max_epochs=100)
        # Set initial best loss
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)

        # No improvement should increment counter
        worse_loss = 0.81
        assert not criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=worse_loss)
        assert criterion.epochs_without_improvement == 1

        even_worse_loss = 0.82
        assert not criterion.should_stop(epoch=2, train_accuracy=0.65, train_loss=even_worse_loss)
        assert criterion.epochs_without_improvement == 2

    def test_stops_after_patience_exceeded(self):
        """Test that PlateauCriterion stops after patience is exceeded."""
        patience = 2
        criterion = PlateauCriterion(patience=patience, min_delta=0.01, max_epochs=100)

        # Set initial best loss
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)

        # First epoch without improvement
        assert not criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=0.81)

        # Second epoch without improvement should trigger stop
        assert criterion.should_stop(epoch=2, train_accuracy=0.65, train_loss=0.82)

    def test_respects_min_delta_threshold(self):
        """Test that PlateauCriterion respects min_delta threshold."""
        min_delta = 0.05
        criterion = PlateauCriterion(patience=3, min_delta=min_delta, max_epochs=100)

        # Set initial best loss
        initial_loss = HIGH_LOSS
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=initial_loss)

        # Small improvement (less than min_delta) should not reset counter
        small_improvement = initial_loss - 0.04  # Less than min_delta
        assert not criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=small_improvement)
        assert criterion.epochs_without_improvement == 1

        # Large improvement (more than min_delta) should reset counter
        large_improvement = initial_loss - 0.1  # More than min_delta
        assert not criterion.should_stop(epoch=2, train_accuracy=0.7, train_loss=large_improvement)
        assert criterion.epochs_without_improvement == 0
        assert criterion.best_train_loss == large_improvement

    def test_reset_restores_initial_state(self):
        """Test that PlateauCriterion reset() properly resets internal state."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01, max_epochs=100)

        # Change state
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)
        criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=0.81)

        # Verify state has changed
        assert criterion.best_train_loss == HIGH_LOSS
        assert criterion.epochs_without_improvement == 1

        # Reset should restore initial state
        criterion.reset()
        assert criterion.best_train_loss == float("inf")
        assert criterion.epochs_without_improvement == 0

    def test_ignores_validation_metrics(self):
        """Test that PlateauCriterion ignores validation metrics."""
        criterion = PlateauCriterion(patience=3, min_delta=0.01, max_epochs=100)

        result_without_val = criterion.should_stop(
            epoch=0, train_accuracy=TYPICAL_ACCURACY, train_loss=TYPICAL_LOSS
        )

        criterion.reset()
        result_with_val = criterion.should_stop(
            epoch=0,
            train_accuracy=TYPICAL_ACCURACY,
            train_loss=TYPICAL_LOSS,
            val_accuracy=0.6,
            val_loss=0.5,
        )
        assert result_without_val == result_with_val

    def test_stop_pattern_over_plateauing_sequence(self, training_sequence):
        """Full decision trace over a realistic improve-then-plateau sequence."""
        criterion = PlateauCriterion(patience=3, min_delta=0.02, max_epochs=100)

        results = [
            criterion.should_stop(epoch=epoch, train_accuracy=accuracy, train_loss=loss)
            for epoch, accuracy, loss in training_sequence
        ]

        # Stops once patience (3) epochs pass without an improvement >= min_delta.
        assert results == [False, False, False, False, True, True]

    def test_zero_patience_stops_at_first_non_improvement(self):
        criterion = PlateauCriterion(patience=0, min_delta=0.01, max_epochs=100)

        assert criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)

    def test_tiny_min_delta_still_recognizes_improvement(self):
        """An improvement larger than even a tiny min_delta resets the patience counter."""
        criterion = PlateauCriterion(patience=2, min_delta=0.0001, max_epochs=100)
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)

        improvement_above_delta = HIGH_LOSS - 0.0002
        assert not criterion.should_stop(
            epoch=1, train_accuracy=0.6, train_loss=improvement_above_delta
        )
        assert criterion.epochs_without_improvement == 0

    def test_loss_equality_behavior(self):
        """Test PlateauCriterion behavior when loss stays exactly the same."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01, max_epochs=100)

        # Set initial loss
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=HIGH_LOSS)

        # Exactly same loss should count as no improvement
        assert not criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=HIGH_LOSS)
        assert criterion.epochs_without_improvement == 1

        # Still same loss should trigger stop after patience
        assert criterion.should_stop(epoch=2, train_accuracy=0.65, train_loss=HIGH_LOSS)
        assert criterion.epochs_without_improvement == 2

    def test_increasing_loss_behavior(self):
        """Test PlateauCriterion behavior when loss consistently increases."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01, max_epochs=100)

        # Set initial loss
        initial_loss = 0.5
        criterion.should_stop(epoch=0, train_accuracy=LOW_ACCURACY, train_loss=initial_loss)

        # Increasing loss should count as no improvement
        assert not criterion.should_stop(epoch=1, train_accuracy=0.6, train_loss=0.6)
        assert criterion.epochs_without_improvement == 1

        assert criterion.should_stop(epoch=2, train_accuracy=0.65, train_loss=0.7)
        assert criterion.epochs_without_improvement == 2


class TestPlateauCriterionBestModelSaving:
    """Test best model saving behavior for PlateauCriterion."""

    def test_saves_best_model_on_loss_improvement(self):
        """Test that PlateauCriterion saves model when loss improves."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01)
        criterion.reset()

        # First epoch - should save (first epoch is always best)
        should_stop = criterion.should_stop(0, 0.8, 1.0)
        assert not should_stop
        assert criterion.should_update_best_model()

        # Update model
        model_state1 = {"epoch": 1, "loss": 1.0}
        criterion.update_best_model(model_state1)
        assert criterion.get_best_model_state() == model_state1

        # Second epoch - better loss, should save
        should_stop = criterion.should_stop(1, 0.85, 0.8)
        assert not should_stop
        assert criterion.should_update_best_model()

        # Update model
        model_state2 = {"epoch": 2, "loss": 0.8}
        criterion.update_best_model(model_state2)
        assert criterion.get_best_model_state() == model_state2

        # Third epoch - worse loss, should not save
        should_stop = criterion.should_stop(2, 0.9, 0.9)
        assert not should_stop
        assert not criterion.should_update_best_model()

        # Best model should still be from epoch 2
        assert criterion.get_best_model_state() == model_state2

    def test_saves_best_model_at_perfect_accuracy(self):
        """Test that PlateauCriterion saves model when hitting 100% accuracy."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01)
        criterion.reset()

        # Regular epoch
        criterion.should_stop(0, 0.8, 1.0)
        criterion.update_best_model({"epoch": 1})

        # Hit perfect accuracy - should save and stop
        should_stop = criterion.should_stop(1, 1.0, 0.5)
        assert should_stop
        assert criterion.should_update_best_model()

        # Update model
        perfect_model = {"epoch": 2, "accuracy": 1.0}
        criterion.update_best_model(perfect_model)
        assert criterion.get_best_model_state() == perfect_model

    def test_respects_min_delta_for_best_model(self):
        """Test that min_delta is respected for best model saving."""
        criterion = PlateauCriterion(patience=3, min_delta=0.1)
        criterion.reset()

        # First epoch
        criterion.should_stop(0, 0.8, 1.0)
        criterion.update_best_model({"epoch": 1, "loss": 1.0})

        # Small improvement (below min_delta) - should not save
        should_stop = criterion.should_stop(1, 0.85, 0.95)
        assert not should_stop
        assert not criterion.should_update_best_model()

        # Large improvement (above min_delta) - should save
        should_stop = criterion.should_stop(2, 0.9, 0.8)
        assert not should_stop
        assert criterion.should_update_best_model()

        # Update model
        better_model = {"epoch": 3, "loss": 0.8}
        criterion.update_best_model(better_model)
        assert criterion.get_best_model_state() == better_model

    def test_best_model_when_patience_exhausted(self):
        """Test that best model is preserved when patience is exhausted."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01)
        criterion.reset()

        # Build up to patience exhaustion
        losses = [1.0, 0.5, 0.6, 0.7, 0.8]  # Best at epoch 2
        best_model = None

        for epoch, loss in enumerate(losses):
            should_stop = criterion.should_stop(epoch, 0.7, loss)

            if criterion.should_update_best_model():
                model_state = {"epoch": epoch + 1, "loss": loss}
                criterion.update_best_model(model_state)
                best_model = model_state

            if should_stop:
                break

        # Should have stopped with best model from epoch 2
        assert best_model is not None
        assert best_model["epoch"] == 2
        assert best_model["loss"] == 0.5
        assert criterion.get_best_model_state() == best_model

    def test_best_model_state_cleared_on_reset(self):
        """Test that reset() clears the best model state."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01)
        criterion.reset()

        # Save a model
        criterion.should_stop(0, 0.8, 1.0)
        criterion.update_best_model({"saved": "state"})
        assert criterion.get_best_model_state() is not None

        # Reset should clear everything
        criterion.reset()
        assert criterion.get_best_model_state() is None
        assert criterion.best_train_loss == float("inf")
        assert criterion.epochs_without_improvement == 0

    def test_should_update_best_model_flag_behavior(self):
        """Test that should_update_best_model flag behaves correctly."""
        criterion = PlateauCriterion(patience=2, min_delta=0.01)
        criterion.reset()

        # First epoch - should update
        criterion.should_stop(0, 0.8, 1.0)
        assert criterion.should_update_best_model()

        # After update, flag should be reset
        criterion.update_best_model({"state": "data"})
        assert not criterion.should_update_best_model()

        # Better loss - should update again
        criterion.should_stop(1, 0.85, 0.8)
        assert criterion.should_update_best_model()

        # After update, flag should be reset again
        criterion.update_best_model({"state": "data2"})
        assert not criterion.should_update_best_model()

        # Worse loss - should not update
        criterion.should_stop(2, 0.9, 0.9)
        assert not criterion.should_update_best_model()

    def test_best_model_with_complex_loss_sequence(self):
        """Test best model saving with complex loss progression."""
        criterion = PlateauCriterion(patience=3, min_delta=0.05)
        criterion.reset()

        # Complex loss sequence with multiple improvements
        loss_sequence = [
            (0, 0.8, 1.0),  # First epoch - save
            (1, 0.85, 0.8),  # Better - save
            (2, 0.9, 0.7),  # Better - save
            (3, 0.9, 0.75),  # Worse - don't save
            (4, 0.9, 0.6),  # Better - save
            (5, 0.9, 0.65),  # Worse - don't save
            (6, 0.9, 0.70),  # Worse - don't save
            (7, 0.9, 0.75),  # Worse - don't save (should stop)
        ]

        saved_models = []
        for epoch, accuracy, loss in loss_sequence:
            should_stop = criterion.should_stop(epoch, accuracy, loss)

            if criterion.should_update_best_model():
                model_state = {"epoch": epoch + 1, "loss": loss}
                criterion.update_best_model(model_state)
                saved_models.append(model_state)

            if should_stop:
                break

        # Should have saved models from epochs 1, 2, 3, 5
        assert len(saved_models) == 4
        assert saved_models[0]["epoch"] == 1
        assert saved_models[1]["epoch"] == 2
        assert saved_models[2]["epoch"] == 3
        assert saved_models[3]["epoch"] == 5

        # Best model should be from epoch 5 (loss 0.6)
        best_model = criterion.get_best_model_state()
        assert best_model is not None
        assert best_model["epoch"] == 5
        assert best_model["loss"] == 0.6


class TestBestModelSavingIntegration:
    """Integration tests for best model saving across different criteria."""

    def test_different_criteria_have_different_best_model_logic(self):
        """Test that different criteria have different best model saving logic."""
        accuracy_criterion = AccuracyCriterion(max_epochs=10)
        plateau_criterion = PlateauCriterion(patience=2, min_delta=0.01)

        # Reset both
        accuracy_criterion.reset()
        plateau_criterion.reset()

        # Same epoch data, different behavior
        epoch_data = (0, 0.8, 0.5)  # 80% accuracy, 0.5 loss

        # AccuracyCriterion should not save (no perfect accuracy)
        accuracy_criterion.should_stop(*epoch_data)
        assert not accuracy_criterion.should_update_best_model()

        # PlateauCriterion should save (first epoch is always best)
        plateau_criterion.should_stop(*epoch_data)
        assert plateau_criterion.should_update_best_model()

        # Test perfect accuracy case
        perfect_data = (1, 1.0, 0.3)  # 100% accuracy

        # AccuracyCriterion should save (perfect accuracy)
        accuracy_criterion.should_stop(*perfect_data)
        assert accuracy_criterion.should_update_best_model()

        # PlateauCriterion should also save (better loss)
        plateau_criterion.should_stop(*perfect_data)
        assert plateau_criterion.should_update_best_model()

    def test_model_state_isolation(self):
        """Test that different criteria maintain separate model states."""
        criterion1 = PlateauCriterion(patience=2, min_delta=0.01)
        criterion2 = PlateauCriterion(patience=3, min_delta=0.05)

        criterion1.reset()
        criterion2.reset()

        # Save different models in each criterion
        criterion1.should_stop(0, 0.5, 1.0)
        criterion1.update_best_model({"criterion": 1, "data": "first"})

        criterion2.should_stop(0, 0.6, 0.8)
        criterion2.update_best_model({"criterion": 2, "data": "second"})

        # Each should maintain its own state
        assert criterion1.get_best_model_state() == {"criterion": 1, "data": "first"}
        assert criterion2.get_best_model_state() == {"criterion": 2, "data": "second"}

        # Resetting one should not affect the other
        criterion1.reset()
        assert criterion1.get_best_model_state() is None
        assert criterion2.get_best_model_state() == {"criterion": 2, "data": "second"}


class TestStoppingCriteriaIntegration:
    """Integration tests for stopping criteria in realistic scenarios."""

    def test_strategies_have_different_behavior(self, training_sequence):
        """Test both criteria on the same training scenario to verify different behavior."""
        accuracy_criterion = AccuracyCriterion(max_epochs=10)
        plateau_criterion = PlateauCriterion(patience=3, min_delta=0.01, max_epochs=10)

        # Extend training sequence for more epochs
        extended_sequence = [
            *training_sequence,
            (6, 0.95, 0.694),
            (7, 0.98, 0.695),
            (8, 0.99, 0.696),
            (9, 0.999, 0.697),
        ]

        accuracy_results = []
        plateau_results = []

        for epoch, accuracy, loss in extended_sequence:
            accuracy_stop = accuracy_criterion.should_stop(
                epoch=epoch, train_accuracy=accuracy, train_loss=loss
            )
            plateau_stop = plateau_criterion.should_stop(
                epoch=epoch, train_accuracy=accuracy, train_loss=loss
            )

            accuracy_results.append(accuracy_stop)
            plateau_results.append(plateau_stop)

        # Accuracy criterion should only stop at max epochs (epoch 9)
        assert accuracy_results == [False] * 9 + [True]

        # Plateau criterion should stop earlier when loss stops improving
        assert plateau_results[:6] == [False, False, False, False, False, True]


class TestPlateauCriterionAccuracyFloor:
    """Test PlateauCriterion with accuracy_floor parameter."""

    def test_accuracy_floor_defaults_to_none(self):
        """Test that accuracy_floor defaults to None."""
        criterion = PlateauCriterion()
        assert criterion.accuracy_floor is None

    def test_accuracy_floor_accepts_custom_value(self):
        """Test that accuracy_floor accepts custom values."""
        criterion = PlateauCriterion(accuracy_floor=0.95)
        assert criterion.accuracy_floor == 0.95

    def test_without_accuracy_floor_behaves_normally(self):
        """Test that without accuracy_floor, plateau detection works normally."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Should count patience regardless of accuracy
        assert not criterion.should_stop(0, 0.5, 1.0)  # Initial, save best
        assert not criterion.should_stop(1, 0.6, 1.1)  # Loss worse, patience=1
        assert criterion.should_stop(2, 0.7, 1.2)  # Loss worse, patience=2, stop

    def test_with_accuracy_floor_only_counts_patience_above_threshold(self):
        """Test that with accuracy_floor, patience only counts when accuracy >= threshold."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=0.9)

        # Below threshold: patience should not count
        assert not criterion.should_stop(0, 0.5, 1.0)  # Initial, save best
        assert not criterion.should_stop(1, 0.6, 1.1)  # Loss worse, but accuracy < 0.9, no patience
        assert not criterion.should_stop(2, 0.7, 1.2)  # Loss worse, but accuracy < 0.9, no patience
        assert criterion.epochs_without_improvement == 0  # Should be reset each time

        # Above threshold: patience should count
        assert not criterion.should_stop(3, 0.95, 1.0)  # Above threshold, save best
        assert criterion.should_stop(4, 0.96, 1.1)  # Loss worse, patience=1
        assert criterion.should_stop(5, 0.97, 1.2)  # Loss worse, patience=2, stop

    def test_accuracy_floor_resets_patience_when_falling_below_threshold(self):
        """Test that patience resets when accuracy falls below threshold."""
        criterion = PlateauCriterion(patience=3, accuracy_floor=0.9)

        # Build up patience above threshold
        assert not criterion.should_stop(0, 0.95, 1.0)  # Above threshold, save best
        assert not criterion.should_stop(1, 0.96, 1.1)  # Loss worse, patience=1
        assert not criterion.should_stop(2, 0.97, 1.2)  # Loss worse, patience=2
        assert criterion.epochs_without_improvement == 2

        # Drop below threshold: patience should reset
        assert not criterion.should_stop(3, 0.85, 1.3)  # Below threshold, patience reset
        assert criterion.epochs_without_improvement == 0

        # Back above threshold: patience should start counting again
        assert not criterion.should_stop(4, 0.92, 0.9)  # Above threshold, save best
        assert not criterion.should_stop(5, 0.93, 1.0)  # Loss worse, patience=1
        assert not criterion.should_stop(6, 0.94, 1.1)  # Loss worse, patience=2
        assert criterion.should_stop(7, 0.95, 1.2)  # Loss worse, patience=3, stop

    def test_perfect_accuracy_stops_immediately_regardless_of_floor(self):
        """Test that 100% accuracy stops immediately regardless of accuracy_floor."""
        criterion = PlateauCriterion(accuracy_floor=0.9)

        # Should stop immediately at 100% accuracy
        assert criterion.should_stop(0, 1.0, 0.5)

    def test_accuracy_floor_with_loss_improvement(self):
        """Test that loss improvement resets patience even with accuracy_floor."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=0.9)

        # Above threshold with loss improvement
        assert not criterion.should_stop(0, 0.95, 1.0)  # Above threshold, save best
        assert not criterion.should_stop(1, 0.96, 1.1)  # Loss worse, patience=1
        assert not criterion.should_stop(2, 0.97, 0.9)  # Loss improved, patience reset
        assert criterion.epochs_without_improvement == 0

        # Patience should start counting again
        assert not criterion.should_stop(3, 0.98, 1.0)  # Loss worse, patience=1
        assert criterion.should_stop(4, 0.99, 1.1)  # Loss worse, patience=2, stop

    def test_accuracy_exactly_at_floor_counts_patience(self):
        """The floor comparison is inclusive: accuracy == floor counts patience."""
        criterion = PlateauCriterion(patience=1, accuracy_floor=0.9)

        assert not criterion.should_stop(0, 0.9, 1.0)
        assert criterion.should_stop(1, 0.9, 1.1)  # loss worse at the floor -> patience -> stop

    def test_accuracy_just_below_floor_does_not_count_patience(self):
        criterion = PlateauCriterion(patience=1, accuracy_floor=0.9)

        assert not criterion.should_stop(0, 0.899, 1.0)
        assert not criterion.should_stop(1, 0.899, 1.1)  # loss worse, but below floor
        assert criterion.epochs_without_improvement == 0

    def test_accuracy_floor_with_zero_threshold(self):
        """Test accuracy_floor with 0.0 threshold (should always count patience)."""
        criterion = PlateauCriterion(patience=1, accuracy_floor=0.0)

        # Even very low accuracy should count patience
        assert not criterion.should_stop(0, 0.1, 1.0)  # Save best
        assert criterion.should_stop(1, 0.1, 1.1)  # Loss worse, patience=1, stop

    def test_accuracy_floor_with_high_threshold(self):
        """Test accuracy_floor with very high threshold (0.99)."""
        criterion = PlateauCriterion(patience=1, accuracy_floor=0.99)

        # High accuracy but below threshold
        assert not criterion.should_stop(0, 0.98, 1.0)  # Below threshold
        assert not criterion.should_stop(
            1, 0.98, 1.1
        )  # Loss worse, but below threshold, no patience
        assert criterion.epochs_without_improvement == 0

        # Above threshold (but loss improved significantly, so saved as best)
        assert not criterion.should_stop(
            2, 0.995, 0.9
        )  # Above threshold, save best (loss improved)
        assert criterion.should_stop(3, 0.995, 1.0)  # Loss worse, patience=1, stop

    def test_accuracy_floor_full_sequence_counts_patience_only_above_floor(self):
        """Patience accumulates only after accuracy crosses the floor, so the stop lands
        exactly patience epochs after the crossing — not after the first plateau."""
        criterion = PlateauCriterion(patience=3, accuracy_floor=0.85)

        # Training sequence: low accuracy -> high accuracy -> stuck
        training_sequence = [
            (0, 0.6, 2.0),  # Below threshold
            (1, 0.7, 1.8),  # Below threshold, loss improved
            (2, 0.8, 1.9),  # Below threshold, loss worse but no patience
            (3, 0.9, 1.0),  # Above threshold, save best
            (4, 0.91, 1.1),  # Above threshold, loss worse, patience=1
            (5, 0.92, 1.2),  # Above threshold, loss worse, patience=2
            (6, 0.93, 1.3),  # Above threshold, loss worse, patience=3, should stop
        ]

        for epoch, accuracy, loss in training_sequence[:-1]:
            assert not criterion.should_stop(epoch, accuracy, loss)

        # Final epoch should trigger stop
        epoch, accuracy, loss = training_sequence[-1]
        assert criterion.should_stop(epoch, accuracy, loss)

    def test_accuracy_floor_reset_behavior(self):
        """Test that reset() clears accuracy_floor state correctly."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=0.9)

        # Build up some state
        criterion.should_stop(0, 0.95, 1.0)
        criterion.should_stop(1, 0.96, 1.1)
        assert criterion.epochs_without_improvement == 1

        # Reset should clear state
        criterion.reset()
        assert criterion.epochs_without_improvement == 0
        assert criterion.best_train_loss == float("inf")
        assert criterion.accuracy_floor == 0.9  # Should preserve accuracy_floor value

    def test_accuracy_floor_best_model_tracking(self):
        """Test that best model tracking works correctly with accuracy_floor."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=0.9)

        # Below threshold: should still track best model
        assert not criterion.should_stop(0, 0.8, 1.0)
        assert criterion.should_update_best_model()
        criterion.update_best_model({"epoch": 1, "loss": 1.0})

        assert not criterion.should_stop(1, 0.85, 0.9)  # Better loss, below threshold
        assert criterion.should_update_best_model()
        criterion.update_best_model({"epoch": 2, "loss": 0.9})

        # Above threshold: should continue tracking best model
        assert not criterion.should_stop(2, 0.92, 0.8)  # Better loss, above threshold
        assert criterion.should_update_best_model()
        criterion.update_best_model({"epoch": 3, "loss": 0.8})

        # Best model should be the one with best loss
        best_model = criterion.get_best_model_state()
        assert best_model is not None
        assert best_model["epoch"] == 3
        assert best_model["loss"] == 0.8


class TestStoppingCriterionFactory:
    """Test the create_stopping_criterion factory function."""

    def test_create_plateau_criterion(self):
        """Test factory creates PlateauCriterion with correct parameters."""
        from audioloop.config import AudioLoopConfig
        from audioloop.utils.stopping_criteria import PlateauCriterion, create_stopping_criterion

        config = AudioLoopConfig(
            stopping_criterion_type="plateau",
            max_epochs=800,
            patience=25,
            min_delta=0.02,
            accuracy_floor=0.85,
        )

        criterion = create_stopping_criterion(config)

        assert isinstance(criterion, PlateauCriterion)
        assert criterion.max_epochs == 800
        assert criterion.patience == 25
        assert criterion.min_delta == 0.02
        assert criterion.accuracy_floor == 0.85

    def test_create_accuracy_criterion(self):
        """Test factory creates AccuracyCriterion with correct parameters."""
        from audioloop.config import AudioLoopConfig
        from audioloop.utils.stopping_criteria import AccuracyCriterion, create_stopping_criterion

        config = AudioLoopConfig(stopping_criterion_type="accuracy", max_epochs=1200)

        criterion = create_stopping_criterion(config)

        assert isinstance(criterion, AccuracyCriterion)
        assert criterion.max_epochs == 1200

    def test_create_criterion_invalid_type(self):
        """Test factory raises error for invalid criterion type."""
        from audioloop.config import AudioLoopConfig
        from audioloop.utils.stopping_criteria import create_stopping_criterion

        config = AudioLoopConfig()
        config.stopping_criterion_type = "invalid_type"  # Bypass validation

        with pytest.raises(ValueError, match="Unknown stopping criterion: invalid_type"):
            create_stopping_criterion(config)

    def test_create_criterion_uses_config_defaults(self):
        """Test factory uses config defaults when not explicitly set."""
        from audioloop.config import AudioLoopConfig
        from audioloop.utils.stopping_criteria import PlateauCriterion, create_stopping_criterion

        config = AudioLoopConfig()  # Use all defaults
        criterion = create_stopping_criterion(config)

        assert isinstance(criterion, PlateauCriterion)  # Default type is plateau
        assert criterion.max_epochs == 1000  # Config default
        assert criterion.patience == 20  # Config default
        assert criterion.min_delta == 0.01  # Config default
        assert criterion.accuracy_floor is None  # Config default


class MockTrainingDataset:
    """Mock training dataset for testing auto-accuracy floor calculation."""

    def __init__(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"label": self.labels[idx]}


class TestPlateauCriterionAutoAccuracyFloor:
    """Test auto-calculation of accuracy floor in PlateauCriterion."""

    def test_auto_accuracy_floor_balanced_dataset(self):
        """Test auto-calculation with balanced dataset (50/50)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Balanced dataset: 50% positive, 50% negative
        balanced_dataset = MockTrainingDataset([0] * 50 + [1] * 50)

        # First call should trigger auto-calculation
        should_stop = criterion.should_stop(0, 0.6, 0.5, train_dataset=balanced_dataset)

        # Auto-calculated floor should be max(0.5, 0.5) + 0.15 = 0.65
        assert criterion.accuracy_floor == 0.65
        assert should_stop is False

    def test_auto_accuracy_floor_imbalanced_dataset_majority_negative(self):
        """Test auto-calculation with imbalanced dataset (10% positive)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Imbalanced dataset: 10% positive, 90% negative
        imbalanced_dataset = MockTrainingDataset([0] * 90 + [1] * 10)

        # First call should trigger auto-calculation
        criterion.should_stop(0, 0.8, 0.5, train_dataset=imbalanced_dataset)

        # Auto-calculated floor should be max(0.1, 0.9) + 0.15 = 0.9 + 0.15 = 0.99 (capped)
        assert criterion.accuracy_floor == 0.99

    def test_auto_accuracy_floor_imbalanced_dataset_majority_positive(self):
        """Test auto-calculation with imbalanced dataset (90% positive)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Imbalanced dataset: 90% positive, 10% negative
        imbalanced_dataset = MockTrainingDataset([1] * 90 + [0] * 10)

        # First call should trigger auto-calculation
        criterion.should_stop(0, 0.8, 0.5, train_dataset=imbalanced_dataset)

        # Auto-calculated floor should be max(0.9, 0.1) + 0.15 = 0.9 + 0.15 = 0.99 (capped)
        assert criterion.accuracy_floor == 0.99

    def test_auto_accuracy_floor_extreme_imbalance(self):
        """Test auto-calculation with extreme imbalance (1% positive)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Extremely imbalanced: 1% positive, 99% negative
        extreme_dataset = MockTrainingDataset([0] * 99 + [1] * 1)

        criterion.should_stop(0, 0.8, 0.5, train_dataset=extreme_dataset)

        # Auto-calculated floor should be max(0.01, 0.99) + 0.15 = 0.99 + 0.15 = 0.99 (capped)
        assert criterion.accuracy_floor == 0.99

    def test_auto_accuracy_floor_moderate_imbalance(self):
        """Test auto-calculation with moderate imbalance (25% positive)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Moderately imbalanced: 25% positive, 75% negative
        moderate_dataset = MockTrainingDataset([0] * 75 + [1] * 25)

        criterion.should_stop(0, 0.8, 0.5, train_dataset=moderate_dataset)

        # Auto-calculated floor should be max(0.25, 0.75) + 0.15 = 0.75 + 0.15 = 0.90
        assert criterion.accuracy_floor == 0.90

    def test_auto_accuracy_floor_only_calculated_once(self):
        """Test that auto-calculation only happens once."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # First dataset
        dataset1 = MockTrainingDataset([0] * 75 + [1] * 25)  # 25% positive
        criterion.should_stop(0, 0.8, 0.5, train_dataset=dataset1)
        first_floor = criterion.accuracy_floor

        # Second call with different dataset should not recalculate
        dataset2 = MockTrainingDataset([0] * 50 + [1] * 50)  # 50% positive
        criterion.should_stop(1, 0.8, 0.5, train_dataset=dataset2)

        # Floor should remain the same (from first calculation)
        assert criterion.accuracy_floor == first_floor
        assert criterion.accuracy_floor == 0.90  # From first dataset

    def test_auto_accuracy_floor_no_dataset_provided(self):
        """Test behavior when no train_dataset is provided in kwargs."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Call without train_dataset - should not crash and floor should remain None
        should_stop = criterion.should_stop(0, 0.8, 0.5)

        assert criterion.accuracy_floor is None
        assert should_stop is False

    def test_auto_accuracy_floor_empty_dataset(self):
        """Test auto-calculation with empty dataset (edge case)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        empty_dataset = MockTrainingDataset([])
        criterion.should_stop(0, 0.8, 0.5, train_dataset=empty_dataset)

        # Empty dataset should default to balanced (0.5), so floor = 0.5 + 0.15 = 0.65
        assert criterion.accuracy_floor == 0.65

    def test_auto_accuracy_floor_single_class_all_positive(self):
        """Test auto-calculation with single class (all positive)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        all_positive_dataset = MockTrainingDataset([1] * 100)
        criterion.should_stop(0, 0.8, 0.5, train_dataset=all_positive_dataset)

        # All positive: trivial accuracy = max(1.0, 0.0) = 1.0, floor = min(1.0 + 0.15, 0.99) = 0.99
        assert criterion.accuracy_floor == 0.99

    def test_auto_accuracy_floor_single_class_all_negative(self):
        """Test auto-calculation with single class (all negative)."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        all_negative_dataset = MockTrainingDataset([0] * 100)
        criterion.should_stop(0, 0.8, 0.5, train_dataset=all_negative_dataset)

        # All negative: trivial accuracy = max(0.0, 1.0) = 1.0, floor = min(1.0 + 0.15, 0.99) = 0.99
        assert criterion.accuracy_floor == 0.99

    def test_auto_accuracy_floor_manual_override_preserved(self):
        """Test that manually set accuracy_floor is not overridden."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=0.8)  # Manual value

        dataset = MockTrainingDataset([0] * 90 + [1] * 10)  # Would auto-calc to 0.99
        criterion.should_stop(0, 0.8, 0.5, train_dataset=dataset)

        # Manual value should be preserved
        assert criterion.accuracy_floor == 0.8

    def test_auto_accuracy_floor_reset_behavior(self):
        """Test that reset preserves auto-calculation behavior."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # First calculation
        dataset1 = MockTrainingDataset([0] * 75 + [1] * 25)
        criterion.should_stop(0, 0.8, 0.5, train_dataset=dataset1)
        assert criterion.accuracy_floor == 0.90

        # Reset should restore auto-calculation capability
        criterion.reset()
        assert criterion.accuracy_floor is None
        assert hasattr(criterion, "_auto_accuracy_floor")
        assert criterion._auto_accuracy_floor is True

        # Should auto-calculate again with new dataset
        dataset2 = MockTrainingDataset([0] * 50 + [1] * 50)
        criterion.should_stop(0, 0.8, 0.5, train_dataset=dataset2)
        assert criterion.accuracy_floor == 0.65  # New calculation

    def test_auto_accuracy_floor_prevents_early_stopping(self):
        """Test that auto-calculated floor prevents stopping at pathological accuracy."""
        criterion = PlateauCriterion(patience=2, accuracy_floor=None)

        # Dataset with 75% negative examples (like the brass instrument case)
        dataset = MockTrainingDataset([0] * 75 + [1] * 25)

        # First call auto-calculates floor to 0.90
        should_stop = criterion.should_stop(0, 0.75, 0.5, train_dataset=dataset)
        assert criterion.accuracy_floor == 0.90
        assert should_stop is False  # Accuracy 0.75 < floor 0.90, should continue

        # Subsequent calls at pathological accuracy should not increment patience
        should_stop = criterion.should_stop(1, 0.75, 0.6)  # Worse loss, below floor
        assert should_stop is False  # Should not count toward patience
        assert criterion.epochs_without_improvement == 0  # Patience not incremented

        # Once above floor, patience should start counting
        should_stop = criterion.should_stop(2, 0.91, 0.6)  # Above floor, worse loss
        assert should_stop is False  # Not stopped yet
        assert criterion.epochs_without_improvement == 1  # Now counting patience

        # Second epoch above floor with no improvement should trigger stopping
        should_stop = criterion.should_stop(3, 0.91, 0.6)  # Still above floor, no improvement
        assert should_stop is True  # Should stop (patience=2, this is epoch 2 without improvement)
