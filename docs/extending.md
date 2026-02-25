# Extending AudioLoop

How to add custom models, selection strategies, and stopping criteria. For adding custom datasets, see [custom_datasets.md](custom_datasets.md) instead.

## Custom Models

AudioLoop discovers models automatically by file naming convention, the same way datasets work.

### Quick Start
1. Create `src/audioloop/models/my_model.py`
2. Define a class inheriting from `AudioLoopModel`
3. Implement `forward()`, `get_model_info()`, and `can_handle_shape()`
4. Use immediately: `--model-type my_model`

### Requirements
- **Inherit from `AudioLoopModel`** (extends `nn.Module` with metadata)
- **Accept `num_classes` and `**kwargs`** in constructor
- **Return logits** from `forward()`

### Naming Convention
- **File naming**: `my_model.py` → model name `my_model`
- **Class naming**: Any name — the system finds the `AudioLoopModel` subclass automatically
- **Auto-discovery**: No registration needed

### Custom Parameters
```python
config = AudioLoopConfig(
    model_type="resnet",
    model_kwargs={"depth": 50, "dropout_rate": 0.2}
)
```

Constructor parameters from `get_model_info()` are automatically saved and restored.

For a complete walkthrough with code examples and testing guidance, see [adding_new_models.md](adding_new_models.md).

## Custom Selection Strategies

Selection strategies determine which candidates are presented to the human labeler. AudioLoop uses a Strategy pattern with auto-discovery.

### Quick Start
1. Create `src/audioloop/strategies/my_strategy.py`
2. Define a class inheriting from `SelectionStrategy`
3. Implement `select_candidates(predictions_df, n, **kwargs) -> DataFrame`
4. Use immediately: `--selection-mode my_strategy`

### Built-in Strategies
- **`entropy`** (default): Selects highest-uncertainty examples
- **`confidence`**: Selects highest-confidence examples
- **`mixed_entropy`**: Samples across entropy levels (70% high / 20% medium / 10% low)
- **`basic_transition`**: Starts with confidence, switches to entropy based on performance

See [candidate_selection_explained.md](candidate_selection_explained.md) for detailed explanations of the built-in strategies.

## Custom Stopping Criteria

### Training Stopping (Within a Cycle)
Controls when to stop training epochs for a single model.

1. Inherit from `TrainingStoppingCriterion`
2. Implement `should_stop(epoch, train_accuracy, train_loss) -> bool`
3. Implement `reset()` for state management

See [stopping_criteria_guide.md](stopping_criteria_guide.md) for the full API and examples.

### Cycle Stopping (Across Cycles)
Controls when to stop the active learning loop.

1. Inherit from `CycleStoppingCriterion`
2. Implement `should_stop(current_cycle) -> bool` and `get_best_cycle() -> int`

See [cycle_stopping_criteria.md](cycle_stopping_criteria.md) for details.

## Architecture Notes

All extensibility in AudioLoop follows the same pattern:
- **File-based auto-discovery**: Create a file in the right directory, it's found automatically
- **Strategy pattern**: Abstract base class with pluggable implementations
- **Convention over configuration**: File naming determines the CLI name, no registration step
