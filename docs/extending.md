# Extending AudioLoop

How to add custom models, selection strategies, and stopping criteria. For adding custom datasets, see [custom_datasets.md](custom_datasets.md) instead.

## Custom Models

AudioLoop discovers models automatically by file naming convention, the same way datasets work.

### Quick Start
1. Create `models/my_model.py` in your project root (or `src/audioloop/models/my_model.py` in the package)
2. Define a class inheriting from `AudioLoopModel`
3. Implement `forward()`, `get_model_info()`, and `can_handle_shape()`
4. Use immediately: `--model-type my_model`

> **Project-level models**: The `models/` directory in your project root is the recommended location for custom models — it keeps your code separate from the installed package and survives reinstalls. Project-level models take precedence over built-in models with the same name. The directory is not created by `audioloop init`; create it yourself when needed.

### Requirements
- **Inherit from `AudioLoopModel`** (extends `nn.Module` with metadata)
- **Accept `num_classes` and `**kwargs`** in constructor
- **Return logits** from `forward()`

### Naming Convention
- **File naming**: `my_model.py` → model name `my_model`
- **Class naming**: Any name — the system finds the `AudioLoopModel` subclass automatically
- **Auto-discovery**: No registration needed

### Custom Parameters

Model-specific parameters are passed through `model_kwargs` in YAML or when constructing `AudioLoopConfig` in Python:

```python
config = AudioLoopConfig(
    model_type="resnet",
    model_kwargs={"depth": 50, "dropout_rate": 0.2}
)
```

The CLI currently supports selecting a model with `--model-type`; arbitrary model-specific kwargs are configured through YAML/API rather than command-line flags.

Constructor parameters from `get_model_info()` are automatically saved and restored. Include any architecture-affecting kwargs there so checkpoints load with the same model configuration used during training.

For a complete walkthrough with code examples and testing guidance, see [adding_new_models.md](adding_new_models.md).

## Feature Extractors

Feature extractors turn audio into the tensors models train on. Two ship today — `spectrogram` (log-mel, 2D) and `embedding` (frozen pretrained vectors, 1D) — selected with `feature_extractor_type` in `audioloop.yaml`. See [Feature Extraction](user_manual.md#feature-extraction) for usage.

Unlike models and datasets, extractors are **not** file-discovered. The set is a small explicit dict in `config.py` (`get_feature_extractor`), so adding one means editing that mapping rather than dropping a file in a directory — a deliberate choice while the roster is tiny and each extractor implies a coordinated model (a 2D extractor is useless without a 2D model). To add one:

1. Subclass `FeatureExtractor` (in `src/audioloop/feature_extractor/`). Reuse the base machinery (caching, path resolution, audio loading); implement `extract_one()`, `get_output_shape()`, `cache_subdir`, and `cache_params()`.
2. Add it to the dispatch dict in `config.get_feature_extractor`.
3. Add a model whose `can_handle_shape()` accepts your extractor's output rank (see [Custom Models](#custom-models)).

The `cache_subdir` keeps your extractor's `.pt` files in their own namespace, and `cache_params()` feeds the `extractor.json` staleness check — see [Shape Compatibility](shape_compatibility_and_variable_lengths.md) and the User Manual's Feature Extraction section.

## Custom Selection Strategies

Selection strategies determine which candidates are presented to the human labeler.

### Quick Start
1. Create `src/audioloop/utils/candidate_selection/my_strategy.py`
2. Define a class inheriting from `CandidateSelectionStrategy` (in `base.py`)
3. Implement `select_candidates(predictions, num_candidates, **kwargs) -> list[dict]`
4. Register in the `strategies` dict in `factory.py`
5. Use with: `--selection-mode my_strategy`

### Built-in Strategies
- **`entropy`** (default): Selects highest-uncertainty examples
- **`confidence`**: Selects highest-confidence examples
- **`mixed_entropy`**: Samples across entropy levels (70% high / 20% medium / 10% low)
- **`basic_transition`**: Starts with confidence, switches to entropy based on performance
- **`stratified_uncertainty`**: Samples across stratified uncertainty bins
- **`random`**: Random selection (baseline)

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

- **Models**: File-based auto-discovery — project-level `models/` takes precedence over built-in `audioloop/models/`
- **Feature extractors**: Explicit dict registration in `config.get_feature_extractor` (not file-discovered) — small fixed roster, each paired with a compatible model
- **Selection strategies**: Factory registration — add a class and register it in `factory.py`
- **Stopping criteria**: Instantiated directly in training/cycle configuration code
- **Common pattern**: Abstract base class with pluggable implementations
