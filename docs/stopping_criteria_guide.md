# Stopping Criteria Guide

This guide explains how to use and extend the stopping criteria system in AudioLoop for controlling when model training should stop.

## Overview

AudioLoop uses a pluggable Strategy pattern for training stopping decisions. This allows you to easily switch between different stopping criteria or implement custom ones for specific use cases.

## Built-in Stopping Criteria

### AccuracyCriterion

The simplest stopping criterion that stops when:
- Training accuracy reaches 100%, or
- Maximum number of epochs is reached

```python
from audioloop.utils.stopping_criteria import AccuracyCriterion

# Default: stops at 100% accuracy or 1000 epochs
criterion = AccuracyCriterion()

# Custom max epochs
criterion = AccuracyCriterion(max_epochs=500)

# Usage in training loop
for epoch in range(1000):
    # ... your training code ...
    train_accuracy = compute_accuracy()
    train_loss = compute_loss()
    
    if criterion.should_stop(epoch, train_accuracy, train_loss):
        print(f"Stopping at epoch {epoch}")
        break
```

### PlateauCriterion

A more sophisticated criterion that stops when:
- Training accuracy reaches 100%, or
- Training loss stops improving (plateaus), or
- Maximum number of epochs is reached

```python
from audioloop.utils.stopping_criteria import PlateauCriterion

# Default: patience=50, min_delta=0.01, max_epochs=1000, accuracy_floor=None
criterion = PlateauCriterion()

# Custom parameters
criterion = PlateauCriterion(
    patience=20,        # Wait 20 epochs without improvement
    min_delta=0.005,    # Minimum improvement threshold
    max_epochs=800,     # Maximum epochs
    accuracy_floor=0.95 # Only count patience when accuracy >= 95%
)
```

# Usage
for epoch in range(1000):
    # ... training code ...
    
    if criterion.should_stop(epoch, train_accuracy, train_loss):
        print(f"Stopping at epoch {epoch}")
        print(f"Best loss: {criterion.best_train_loss}")
        print(f"Epochs without improvement: {criterion.epochs_without_improvement}")
        break
```

**Key Parameters:**
- `patience`: Number of epochs to wait without improvement before stopping
- `min_delta`: Minimum change in loss to qualify as improvement
- `max_epochs`: Maximum number of epochs (safety net)
- `accuracy_floor`: Only count patience when accuracy >= this threshold (optional)

**State Management:**
- `best_train_loss`: Tracks the best (lowest) training loss seen
- `epochs_without_improvement`: Counter for epochs without significant improvement (resets when accuracy falls below floor)

**Accuracy Floor Behavior:**
- When `accuracy_floor=None` (default): Standard plateau detection at all accuracy levels
- When `accuracy_floor=0.95`: Only starts counting patience when accuracy >= 95%
- If accuracy drops below floor: Patience counter resets to 0
- Purpose: Prevents premature stopping when model hasn't reached high accuracy yet

## Integration with AudioLoop

The stopping criteria integrate with AudioLoop's training system through the config:

```python
from audioloop.config import AudioLoopConfig
from audioloop.training_core import run_training

# Configure stopping criterion via config
config = AudioLoopConfig(
    stopping_criterion_type="plateau",
    patience=30,
    min_delta=0.01,
    accuracy_floor=0.9,
)

# Training creates the criterion from config automatically
run_training(config, labels_file="training_sets/training_set_v1.csv", version=1)
```

**CLI Usage:**
```bash
# Default plateau behavior
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion plateau

# Enhanced plateau with accuracy floor
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion plateau --accuracy-floor 0.95 --patience 20

# High accuracy threshold (only patient when >= 98%)
python -m audioloop.train training_sets/training_set_v1.csv --accuracy-floor 0.98 --patience 10
```

## Creating Custom Stopping Criteria

### Basic Custom Criterion

```python
from audioloop.utils.stopping_criteria import TrainingStoppingCriterion

class CustomCriterion(TrainingStoppingCriterion):
    def __init__(self, target_accuracy=0.95, max_epochs=1000):
        self.target_accuracy = target_accuracy
        self.max_epochs = max_epochs
    
    def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
        # Stop if target accuracy reached
        if train_accuracy >= self.target_accuracy:
            return True
        
        # Stop if max epochs reached
        return epoch >= self.max_epochs - 1
    
    def reset(self):
        # Reset any internal state (if needed)
        pass
```

### Advanced Custom Criterion with State

```python
class EarlyStoppingCriterion(TrainingStoppingCriterion):
    def __init__(self, patience=10, min_delta=0.01, max_epochs=1000):
        self.patience = patience
        self.min_delta = min_delta
        self.max_epochs = max_epochs
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
    
    def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
        # Always stop at perfect accuracy
        if train_accuracy >= 1.0:
            return True
        
        # Early stopping based on validation loss
        if val_loss is not None:
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.patience:
                return True
        
        # Max epochs fallback
        return epoch >= self.max_epochs - 1
    
    def reset(self):
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
```

### Combined Criteria

```python
class CombinedCriterion(TrainingStoppingCriterion):
    def __init__(self, criteria):
        self.criteria = criteria
    
    def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
        # Stop if ANY criterion says to stop
        return any(
            c.should_stop(epoch, train_accuracy, train_loss, val_accuracy, val_loss)
            for c in self.criteria
        )
    
    def reset(self):
        for c in self.criteria:
            c.reset()

# Usage
combined = CombinedCriterion([
    AccuracyCriterion(max_epochs=1000),
    PlateauCriterion(patience=50, min_delta=0.01),
    EarlyStoppingCriterion(patience=20)
])
```

## Advanced Patterns

### Factory Pattern

AudioLoop uses a config-driven factory for consistent criterion creation:

```python
from audioloop.config import AudioLoopConfig
from audioloop.utils.stopping_criteria import create_stopping_criterion

# Create config with desired stopping criterion
config = AudioLoopConfig(
    stopping_criterion_type="plateau",
    patience=30,
    min_delta=0.005,
    max_epochs=1000
)

# Factory reads all parameters from config
criterion = create_stopping_criterion(config)
```

**Direct Instantiation** (for testing or custom use):
```python
from audioloop.utils.stopping_criteria import PlateauCriterion

# Can still instantiate directly when needed
criterion = PlateauCriterion(
    patience=30,
    min_delta=0.005,
    max_epochs=1000
)
```

### Configuration-Driven Criteria

```python
from audioloop.config import AudioLoopConfig
from audioloop.utils.stopping_criteria import create_stopping_criterion

# All configuration in one place
config = AudioLoopConfig(
    stopping_criterion_type="plateau",
    patience=30,
    min_delta=0.005,
    max_epochs=1000
)

# Factory creates the appropriate criterion
criterion = create_stopping_criterion(config)

# Use in training
from audioloop.training_core import run_training
run_training(config, labels_file="training.csv", version=1)
```

## Best Practices

### 1. Choose the Right Criterion

- **AccuracyCriterion**: Simple tasks, small datasets, or when you want to train to completion
- **PlateauCriterion**: Most common choice for general training
- **Custom EarlyStoppingCriterion**: When you have validation data and want to prevent overfitting

### 2. Parameter Tuning

**PlateauCriterion Parameters:**
- `patience`: Start with 10-20 for small datasets, 50-100 for large ones
- `min_delta`: Start with 0.01, reduce to 0.001 for fine-tuning
- `max_epochs`: Set based on your computational budget
- `accuracy_floor`: Use 0.9-0.95 for audio classification to prevent early stopping before reaching good performance

**Recommended Accuracy Floor Values:**
- `accuracy_floor=0.85`: Good for stable audio datasets
- `accuracy_floor=0.90`: Standard for most audio classification tasks
- `accuracy_floor=0.95`: Conservative, only be patient when model is performing well
- `accuracy_floor=None`: Default plateau behavior (always count patience)

### 3. State Management

Always call `reset()` when starting a new training run:

```python
criterion = PlateauCriterion(patience=30)

# Train first model
for epoch in range(1000):
    if criterion.should_stop(epoch, accuracy, loss):
        break

# Train second model - reset state first!
criterion.reset()
for epoch in range(1000):
    if criterion.should_stop(epoch, accuracy, loss):
        break
```

### 4. Monitoring and Debugging

```python
class VerbosePlateauCriterion(PlateauCriterion):
    def should_stop(self, epoch, train_accuracy, train_loss, val_accuracy=None, val_loss=None):
        result = super().should_stop(epoch, train_accuracy, train_loss, val_accuracy, val_loss)
        
        if epoch % 10 == 0:  # Log every 10 epochs
            print(f"Epoch {epoch}: "
                  f"best_loss={self.best_train_loss:.4f}, "
                  f"epochs_without_improvement={self.epochs_without_improvement}")
        
        if result:
            print(f"Stopping at epoch {epoch} - "
                  f"reason: {'perfect accuracy' if train_accuracy >= 1.0 else 'plateau/max_epochs'}")
        
        return result
```

## Testing Custom Criteria

AudioLoop provides comprehensive test utilities. Here's how to test your custom criterion:

```python
import pytest
from audioloop.utils.stopping_criteria import TrainingStoppingCriterion

def test_custom_criterion():
    criterion = CustomCriterion(target_accuracy=0.9, max_epochs=100)
    
    # Test basic functionality
    assert not criterion.should_stop(0, 0.8, 0.1)
    assert criterion.should_stop(0, 0.95, 0.1)
    assert criterion.should_stop(99, 0.8, 0.1)
    
    # Test reset
    criterion.reset()
    # ... test that state is properly reset
```

## Performance Considerations

- Stopping criteria are called every epoch, so keep them lightweight
- Avoid complex computations in `should_stop()`
- For criteria with heavy validation logic, consider caching results

## Common Issues and Solutions

### Issue: Criterion stops too early
**Solution**: Increase `patience` parameter or decrease `min_delta`

### Issue: Training never stops
**Solution**: Check that `max_epochs` is set appropriately, or that your criterion logic is correct

### Issue: Training stops too early with accuracy_floor
**Solution**: Lower the `accuracy_floor` threshold or increase `patience`

### Issue: Training doesn't stop when expected with accuracy_floor
**Solution**: Check that accuracy is actually reaching the floor threshold, or remove `accuracy_floor` for standard behavior

### Issue: State persists between training runs
**Solution**: Always call `reset()` before starting a new training run

### Issue: Custom criterion not working
**Solution**: Ensure you inherit from `TrainingStoppingCriterion` and implement `should_stop()` correctly

## Examples in AudioLoop

The AudioLoop codebase includes several examples of stopping criteria usage:

1. **Default Training**: Uses `PlateauCriterion` (the default stopping criterion)
2. **Simple Training**: Uses `AccuracyCriterion` for training to 100% accuracy
3. **Custom Workflows**: See `automated_workflow.py` for advanced usage patterns

## API Reference

### TrainingStoppingCriterion (Abstract Base Class)

```python
class TrainingStoppingCriterion(ABC):
    @abstractmethod
    def should_stop(self, epoch: int, train_accuracy: float, train_loss: float, 
                   val_accuracy: float | None = None, val_loss: float | None = None) -> bool:
        """Determine if training should stop."""
    
    def reset(self) -> None:
        """Reset internal state for a new training run."""
```

### AccuracyCriterion

```python
class AccuracyCriterion(TrainingStoppingCriterion):
    def __init__(self, max_epochs: int = 1000):
        """
        Args:
            max_epochs: Maximum number of epochs to train
        """
```

### PlateauCriterion

```python
class PlateauCriterion(TrainingStoppingCriterion):
    def __init__(self, patience: int = 50, min_delta: float = 0.01, max_epochs: int = 1000, accuracy_floor: float | None = None):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            max_epochs: Maximum epochs (fallback safety)
            accuracy_floor: Only count patience when accuracy >= this threshold (optional)
        """
```

## Practical Examples

### Example 1: Audio Classification with Accuracy Floor
```python
# Scenario: Audio classification where we want to be patient early in training
# but stop when stuck at high accuracy

from audioloop.utils.stopping_criteria import PlateauCriterion

# Only count patience when accuracy >= 90%
criterion = PlateauCriterion(
    patience=20,
    accuracy_floor=0.90,
    min_delta=0.01,
    max_epochs=1000
)

# Simulate training
training_sequence = [
    (0, 0.6, 1.5),   # Low accuracy - no patience counted
    (10, 0.8, 1.2),  # Still low - no patience counted  
    (20, 0.92, 1.0), # Above floor - start counting patience
    (25, 0.93, 1.1), # Above floor - patience = 1
    (30, 0.94, 1.2), # Above floor - patience = 2
    # ... continues until patience exceeded or improvement
]

for epoch, accuracy, loss in training_sequence:
    if criterion.should_stop(epoch, accuracy, loss):
        print(f"Stopped at epoch {epoch} with {accuracy:.1%} accuracy")
        break
```

### Example 2: Comparing Standard vs Accuracy Floor
```python
# Standard plateau criterion
standard = PlateauCriterion(patience=10)

# Enhanced with accuracy floor
enhanced = PlateauCriterion(patience=10, accuracy_floor=0.85)

# Same training data
training_data = [
    (0, 0.7, 1.0),   # Standard: counts patience, Enhanced: no patience
    (1, 0.8, 1.1),   # Standard: patience=1, Enhanced: patience=0
    (2, 0.9, 1.0),   # Standard: patience=0 (improved), Enhanced: patience=0
    (3, 0.91, 1.1),  # Standard: patience=1, Enhanced: patience=1
    # Enhanced only starts counting when accuracy >= 0.85
]
```

### Example 3: CLI Usage Patterns
```bash
# For early training cycles (small datasets)
python -m audioloop.train training_sets/training_set_v1.csv --accuracy-floor 0.85

# For established models (larger datasets)  
python -m audioloop.train training_sets/training_set_v3.csv --accuracy-floor 0.95 --patience 30

# For fine-tuning (be very patient at high accuracy)
python -m audioloop.train training_sets/training_set_v5.csv --accuracy-floor 0.98 --patience 50
```

## Further Reading

- [Strategy Pattern](https://en.wikipedia.org/wiki/Strategy_pattern) - Design pattern used in stopping criteria
- [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping) - Technique to prevent overfitting
- [AudioLoop Architecture](../README.md) - Overall system architecture