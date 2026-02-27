# Training Stopping Criteria

Training stopping criteria control when to stop training epochs *within a single cycle*. For stopping the *entire active learning loop* across cycles, see [cycle_stopping_criteria.md](cycle_stopping_criteria.md).

## Built-in Criteria

### Plateau (Default)

Stops training when the loss stops improving. This is the default and recommended option.

How it works:
1. Tracks the best (lowest) training loss seen so far
2. If loss doesn't improve by at least `min_delta` for `patience` consecutive epochs, training stops
3. An accuracy floor prevents premature stopping: patience only counts when accuracy is above the floor
4. Training also stops immediately at 100% accuracy or at `max_epochs`

The accuracy floor is auto-calculated by default: `majority_class_percentage + 15%` (capped at 0.99). This prevents early stopping when the model hasn't learned anything meaningful yet; a model that always predicts the majority class can appear to "plateau" at a high loss.

### Accuracy

A simpler criterion that trains until 100% accuracy or `max_epochs`. No patience or loss tracking.

## Configuration

Set these in `audioloop.yaml`:

```yaml
stopping_criterion_type: plateau   # "plateau" or "accuracy"
patience: 20                       # Epochs without improvement before stopping
min_delta: 0.01                    # Minimum loss decrease to count as improvement
accuracy_floor: null               # null = auto-calculate, or a fixed float (0.0-1.0)
max_epochs: 1000                   # Safety limit
```

Or via CLI:

```bash
python -m audioloop.train training_set_v1.csv --stopping-criterion plateau --patience 20
python -m audioloop.train training_set_v1.csv --accuracy-floor 0.95 --patience 30
```

## When to Adjust

**Training stops too early (model hasn't converged)**
- Increase `patience` (e.g., 30-50 instead of 20)
- Decrease `min_delta` (e.g., 0.005 instead of 0.01)
- Set a higher `accuracy_floor` to keep training longer at lower accuracies

**Training runs too long**
- Decrease `patience` (e.g., 10-15)
- Increase `min_delta`
- Lower `max_epochs`

**First cycle is very slow to stop**
- Normal. Early training sets are small and noisy, so the loss bounces around. The accuracy floor helps here, as it won't start counting patience until accuracy is above the trivial baseline.

## Adding Custom Criteria

Subclass `TrainingStoppingCriterion` and implement `should_stop()`:

```python
from audioloop.utils.stopping_criteria import TrainingStoppingCriterion

class MyCustomCriterion(TrainingStoppingCriterion):
    def __init__(self, max_epochs=1000):
        self.max_epochs = max_epochs
        self.best_model_state = None

    def should_stop(self, epoch, train_accuracy, train_loss,
                    val_accuracy=None, val_loss=None, **kwargs):
        if train_accuracy >= 1.0:
            return True
        return epoch >= self.max_epochs - 1

    def reset(self):
        self.best_model_state = None
```

Register it in `create_stopping_criterion()` in `utils/stopping_criteria.py` to make it available via config. See also [extending.md](extending.md).

## See Also
- [User Manual: Training](user_manual.md#training) — overview of training behavior and options
- [Cycle Stopping Criteria](cycle_stopping_criteria.md) — stopping the active learning loop
