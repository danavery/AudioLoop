# AudioLoop Developer Guide

Developer reference for AudioLoop architecture, patterns, and extensibility. For workflow patterns and configuration, see [docs/user_manual.md](docs/user_manual.md).

## Project Architecture

AudioLoop is an active learning framework for binary audio classification supporting arbitrary audio datasets. It implements a versioned workflow for iterative model improvement through human-in-the-loop labeling. Built-in support includes AudioSet, FSD50K, and UrbanSound8K, with easy extensibility for custom datasets.

### Design Goals

The eventual goal is to enable a user to arrive at the application with only a large unlabeled audio dataset and a small labeled subset of that dataset, and let the system independently create a model that can label the entire dataset with high accuracy. This includes:

- Automatic hyperparameter tuning for the entire training process
- Adaptive early stopping strategies and learning rate scheduling  
- Intelligent candidate selection strategies for human labeling
- Both CLI and web interfaces for training and human labeling

### Development Philosophy

This is a project in rapid development with no production users. Maintaining backwards compatibility during code changes is not required and would unnecessarily complicate the codebase. Focus on clean, maintainable code over backwards compatibility.

## Core Concepts

### Workflow Modes

AudioLoop supports two primary workflow modes designed for different use cases:

**Production Mode** (Default):
- Real-world deployment with truly unlabeled datasets
- No ground truth available
- Manual human labeling required
- Tracks prediction and confidence metrics only

**Evaluation Mode** (Research/Testing):
- Research and testing with known datasets
- Ground truth available for comprehensive evaluation
- Auto-labeling possible for rapid testing
- Full evaluation metrics (F1, precision, recall, accuracy)

Use `--evaluation-mode` with `automated_workflow`, or `--with-ground-truth` with individual commands like `active_learning`.

### Selection Strategies

AudioLoop uses a pluggable strategy pattern for candidate selection:

**EntropyStrategy** (Default):
- Selects samples with highest entropy (most uncertain predictions)
- Best general-purpose strategy for active learning
- Focuses on samples near decision boundaries

**ConfidenceStrategy**:
- Selects samples with highest model confidence scores
- Useful for early training cycles to verify model is learning correctly
- Can lead to redundant selections once the model becomes overconfident

**BasicTransitionStrategy**:
- Automatically switches from confidence to entropy based on model performance
- Uses F1 score, mean confidence, and confidence variance as transition criteria
- Adaptive thresholds based on dataset characteristics

## Architecture Overview

### Core Modules
- **`active_learning.py`**: CLI interface for active learning
- **`active_learning_core.py`**: Core active learning pipeline with candidate selection
- **`train.py`**: CLI interface for model training
- **`training_core.py`**: Core training logic with automatic versioning and pluggable stopping criteria
- **`label_audio.py`**: Terminal-based audio labeling interface with multi-dataset support
- **`merge_labels.py`**: Combines human labels with training sets
- **`create_specs.py`**: Preprocesses audio into spectrograms (optional with lazy generation)
- **`create_subset.py`**: CLI tool for creating training-ready dataset subsets with binary classification labels
- **`prepare_subset_specs.py`**: Creates subset-specific spectrogram directories for efficient remote deployment (hard links or copies)
- **`track_metrics.py`**: Comprehensive metrics tracking and visualization (accuracy, F1, precision, recall, confidence, entropy) across active learning cycles
- **`config.py`**: Unified configuration system coordinating paths, datasets, and experiments
- **`utils/create_bootstrap_set.py`**: Bootstrap training set creation from ground truth (evaluation mode only)
- **`utils/paths.py`**: Path utilities eliminating hardcoded path duplication
- **`utils/stopping_criteria.py`**: Pluggable training stopping criteria using Strategy pattern
- **`utils/candidate_selection.py`**: Pluggable candidate selection strategies using Strategy pattern

### Models
- **`models/audio_loop_model.py`**: Abstract base class (`AudioLoopModel`) defining the pluggable model interface
- **`models/cnn_5layer.py`**: Primary CNN architecture with adaptive pooling
- **`models/simple_cnn.py`**: Alternative lightweight CNN model

AudioLoop uses a pluggable model architecture where all models implement the `AudioLoopModel` abstract base class. This allows easy integration of custom PyTorch models or HuggingFace models while maintaining compatibility with the existing training and inference pipeline.

### Shape Compatibility and Variable Lengths

AudioLoop supports variable length spectrograms and automatic dataset/model compatibility checking. This enables domain-specific feature extraction while preventing runtime failures.

**Key Features:**
- **Shape Compatibility System**: Datasets declare output shapes (`get_output_shape`), models declare input requirements (`can_handle_shape`), and the training pipeline automatically validates compatibility.
- **Variable Length Spectrograms**: Natural audio durations are preserved, with outlier cropping and dynamic batch padding.
- **Implicit Temporal Augmentation**: Training on natural length variations improves model generalization.

**Dataset Configuration:**
Datasets now return `(num_mels, -1)` for their shape, indicating a variable time dimension. The `fixed_length` parameter is used as a maximum for outlier cropping, not for forced padding.

```python
class FSD50KConfig(DatasetConfig):
    fixed_length = 2048  # Used as max length

    def get_output_shape(self) -> tuple[int, ...]:
        return (self.n_mels, -1)  # Variable time dimension

    def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
        # Crops outliers > 2048, but preserves natural lengths of shorter clips
        ...
```

**Model Configuration:**
Models declare their input requirements, which can be flexible.

```python
# CNN with adaptive pooling (can handle any 2D shape)
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return len(shape) == 2

# MLP requiring exact element count
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return math.prod(shape) == self.required_elements
```

**Batch Processing:**
The `variable_length_collate_fn` handles different lengths by padding to the longest in each batch, ensuring within-batch consistency.

This system allows for domain-specific optimizations (e.g., whale calls with fewer frequency bins and longer time windows) while maintaining a robust and flexible training pipeline.

### Data Pipeline
- **`utils/spectrogram_dataset.py`**: Unified dataset loader for multiple CSV formats
- **`utils/data_utils.py`**: Core utilities (device selection, collate functions, entropy)
- **`utils/labeled_dataset.py`**: Legacy dataset implementation
- **`utils/log_normalize.py`**: Spectrogram normalization transform

### Dataset Configuration
- **`datasets/dataset_config.py`**: Abstract base class defining common dataset interface
- **`datasets/fsd50k_config.py`**: FSD50K dataset configuration
- **`datasets/urbansound8k_config.py`**: UrbanSound8K dataset configuration
- **`datasets/registry.py`**: Dynamic dataset discovery system

### Strategy Pattern Infrastructure
- **`utils/stopping_criteria.py`**: Pluggable stopping criteria architecture
  - `TrainingStoppingCriterion`: Abstract base class for stopping decisions
  - `AccuracyCriterion`: Default criterion (stop at 100% accuracy or max epochs)
  - Extensible for early stopping, plateau detection, and other strategies
- **`utils/candidate_selection.py`**: Pluggable candidate selection architecture
  - `CandidateSelectionStrategy`: Abstract base class for selection strategies
  - `ConfidenceStrategy`: Selects samples with highest confidence scores
  - `EntropyStrategy`: Selects samples with highest entropy (uncertainty sampling)
  - Extensible for margin-based, diversity-based, and other selection strategies

### Unified Configuration System
AudioLoop uses a centralized configuration system that coordinates all settings:

```python
from audioloop.config import AudioLoopConfig

# Create configuration with project defaults (recommended for CLI commands)
config = AudioLoopConfig.from_project(experiment_name="my_exp")

# Or create directly (for tests or when project context not needed)
config = AudioLoopConfig(experiment_name="my_exp", dataset="urbansound8k")

# Access all path locations
config.output_dir          # outputs/my_exp/
config.training_sets_dir   # training_sets/my_exp/
config.specs_dir          # data/all_specs/

# Generate versioned file paths
config.get_model_path(1)        # outputs/my_exp/model_v1.pt
config.get_predictions_path(1)  # outputs/my_exp/predictions_v1.csv
config.get_training_set_path(1) # training_sets/my_exp/training_set_v1.csv
```

### Configuration Precedence
AudioLoop follows a layered configuration system with proper precedence:
1. **Explicit constructor parameters** (highest priority)
2. **Experiment config file** (`--config` flag, uses `from_yaml()`)
3. **Project defaults** (`audioloop.yaml` in project root, uses `from_project()`)
4. **Environment variables** (`AUDIOLOOP_*` variables)
5. **Default values** (lowest priority)

**Factory Methods:**
- `AudioLoopConfig.from_project(**overrides)` - Loads project defaults from `audioloop.yaml`, applies overrides. **Used by all CLI commands.**
- `AudioLoopConfig.from_yaml(path, **overrides)` - Loads from specific YAML file, applies overrides
- `AudioLoopConfig(**kwargs)` - Direct construction without project/file defaults

```python
# CLI commands use from_project() to pick up audioloop.yaml defaults
config = AudioLoopConfig.from_project(experiment_name="my_exp")

# Direct construction for tests or explicit configuration
config = AudioLoopConfig(dataset='fsd50k')  # Ignores audioloop.yaml
```

### Versioned Workflow System
AudioLoop uses consistent versioning across all artifacts managed through the configuration system:
- Models: `config.get_model_path(N)` → `outputs[_experiment]/model_vN.pt`
- Training sets: `config.get_training_set_path(N)` → `training_sets[_experiment]/training_set_vN.csv`
- Predictions: `config.get_predictions_path(N)` → `outputs[_experiment]/predictions_vN.csv`
- Candidates: `config.get_candidates_path(N)` → `outputs[_experiment]/labeling_candidates_vN.csv`
- Binary labels: `config.get_binary_labels_path(N)` → `outputs[_experiment]/binary_labels_vN.csv`

### Environment Variables
Customize paths and behavior via environment variables:
- `AUDIOLOOP_DATASET`: Default dataset (`fsd50k` or `urbansound8k`)
- `AUDIOLOOP_DATA_ROOT`: Root directory for data files (default: `data`)
- `AUDIOLOOP_OUTPUT_ROOT`: Root directory for outputs (default: `.`)
- `AUDIOLOOP_SPECS_DIR`: Spectrograms subdirectory (default: `all_specs`). Env var fallback; prefer `specs_dir_path` in yaml.

### Sound Classification
- **`datasets/fsd50k.py`**: FSD50K class mappings (200 classes with semantic groupings)
- **`datasets/urbansound8k.py`**: UrbanSound8K class mappings (10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music)
- Binary classification framework converts any class into positive/negative labels

## Data Flow

### Production Mode (Default)
1. **Subset Creation** (optional, for large datasets): Create manageable subsets via `create_subset.py`
2. **Preprocessing** (optional): Raw audio → spectrograms via `create_specs.py`, or use lazy generation
3. **Initial Training Set**: User-provided labeled dataset
4. **Model Training**: Small labeled set → CNN model via `train.py` (with lazy spec generation if audio_path provided)
5. **Active Learning**: Model predictions on ALL files → candidate selection via `active_learning.py`
6. **Human Labeling**: Audio playback + labeling via `label_audio.py` or web UI
7. **Label Integration**: Human labels → expanded training set via `merge_labels.py`
8. **Performance Monitoring**: Prediction and confidence metrics via `track_metrics.py`
9. **Iteration**: Repeat training with expanded data

### Evaluation Mode (Research/Testing)
1. **Subset Creation** (optional, for large datasets): Create manageable subsets via `create_subset.py`
2. **Preprocessing** (optional): Raw audio → spectrograms via `create_specs.py`, or use lazy generation
3. **Bootstrap Training Set**: Sample from ground truth via `utils/create_bootstrap_set.py`
4. **Model Training**: Small labeled set → CNN model via `train.py` (with lazy spec generation if audio_path provided)
5. **Active Learning**: Model predictions with ground truth → candidate selection via `active_learning.py --with-ground-truth`
6. **Auto-Labeling**: Ground truth extraction via `auto_label_candidates.py` (optional)
7. **Label Integration**: Labels → expanded training set via `merge_labels.py`
8. **Performance Evaluation**: Full ground truth metrics (F1, precision, recall) via `track_metrics.py`
9. **Iteration**: Repeat training with expanded data

**Key Difference**: Evaluation mode includes ground truth data for comprehensive performance analysis, while production mode works with truly unknown data.

### Clean Architecture
- **Active Learning**: Focused inference engine that produces predictions and candidate selections
- **Metrics Tracking**: Comprehensive performance analysis across active learning iterations
- **Candidate Selection**: Uses only model outputs (confidence, entropy) for selection decisions
- **Strategy Pattern**: Consistent pluggable architecture for both training stopping criteria and candidate selection strategies

## File Formats

### Training Set CSV
Bootstrap (from `create_bootstrap_set`):
```csv
filename,label,audio_path
clip_001.pt,1,/path/to/audio/clip_001.wav
```

Merged (from `merge_labels`):
```csv
filename,label
data/all_specs/100032-3-0-0.pt,1
```

### Predictions CSV (Generated)
Production mode:
```csv
filename,prediction,predicted_class,target_class,confidence,entropy,prob_negative,prob_positive,original_class,audio_path,filepath
```

Evaluation mode (--with-ground-truth flag):
```csv
filename,ground_truth,prediction,predicted_class,target_class,confidence,entropy,prob_negative,prob_positive,original_class,correct,audio_path,filepath
```

### Candidates CSV (For Human Labeling)
```csv
filename,prediction,predicted_class,target_class,confidence,entropy,prob_negative,prob_positive,original_class,audio_path,filepath,needs_human_label,human_confidence
```

## Key Dependencies

- **PyTorch**: Neural network training and inference
- **TorchAudio**: Audio processing and spectrogram generation
- **NumPy**: Numerical operations
- **TQDM**: Progress bars
- **Ruff**: Code formatting and linting

## Development Patterns

### Unified Configuration Pattern
All modules use the unified configuration system instead of scattered parameters:
```python
# CLI commands: use from_project() to inherit audioloop.yaml defaults
from audioloop.config import AudioLoopConfig
config = AudioLoopConfig.from_project(experiment_name="test")

# Tests or explicit config: use direct construction
config = AudioLoopConfig(experiment_name="test", dataset="urbansound8k")

# Pass config object to functions
run_training(config, labels_file="training.csv", ...)
run_active_learning_cycle(config, positive_class_name="Drill", ...)
```

### Strategy Pattern Implementation
Both training stopping criteria and candidate selection follow the same Strategy pattern:
- Abstract base class defines the interface
- Concrete strategy classes implement specific algorithms
- Config-driven factory functions for consistent instantiation
- Each strategy has complete control over its domain logic

**Factory Pattern** (Recommended for Production Code):
```python
# Training stopping criteria - factory creates from config
from audioloop.config import AudioLoopConfig
from audioloop.utils.stopping_criteria import create_stopping_criterion

config = AudioLoopConfig(stopping_criterion_type="plateau", patience=50)
criterion = create_stopping_criterion(config)

# Candidate selection strategies - factory creates from config
config = AudioLoopConfig(selection_mode="basic_transition", auto_thresholds=True)
from audioloop.utils.candidate_selection import create_strategy
strategy = create_strategy(config)
```

**Direct Instantiation** (For Tests or Custom Use):
```python
# Can also instantiate strategies directly for testing or custom scenarios
from audioloop.utils.stopping_criteria import PlateauCriterion
from audioloop.utils.candidate_selection import ConfidenceStrategy

criterion = PlateauCriterion(patience=50, min_delta=0.01)
strategy = ConfidenceStrategy()
```

**Why Use Factories:**
- Config validation happens once at config creation
- Encapsulates strategy-specific parameter mapping
- Consistent pattern across all modules
- Single source of truth for configuration

### Class Weighting Configuration
AudioLoop supports three class weighting modes for handling imbalanced datasets:

**Design Rationale**: Active learning with imbalanced classes can cause training set class ratios to drift between cycles, leading to unstable model performance. Class weighting provides consistent training signals across cycles.

**Configuration:**
```python
from audioloop.config import AudioLoopConfig

# Fixed weighting (default) - prevents model collapse with imbalanced training data
config = AudioLoopConfig(class_weighting=0.70)  # Target 70% positive weight

# No weighting - treats all classes equally (can collapse to all-positive predictions)
config = AudioLoopConfig(class_weighting=None)

# Adaptive weighting - calculates inverse frequency from training set each cycle
config = AudioLoopConfig(class_weighting="adaptive")
```

**Implementation**: The `class_weighting` parameter is a unified setting that replaced the previous boolean `use_class_weighting` flag, enabling experimentation with fixed target ratios:
- `float` (0.0-1.0): Fixed weights as `weight_positive = (1.0 - target) / target` (default: 0.70)
- `"adaptive"`: Weights calculated as `total_samples / (num_classes * class_counts)` each cycle
- `None`: Standard CrossEntropyLoss without weights

**When to Use**:
- **Fixed 0.70 (default)**: Recommended for most active learning scenarios; prevents model collapse when training set is imbalanced
- **Adaptive**: When training set composition closely matches real-world distribution
- **No weighting**: Only for naturally balanced datasets or baseline comparisons

### Dataset Extensibility
AudioLoop supports adding custom datasets through file-based auto-discovery. Create a `DatasetConfig` subclass in `src/audioloop/datasets/` and it's available in all CLI commands immediately.

See **[docs/custom_datasets.md](docs/custom_datasets.md)** for the full walkthrough.

### Model Extensibility
AudioLoop supports adding custom models through the same auto-discovery pattern. Create an `AudioLoopModel` subclass in `src/audioloop/models/` and use `--model-type my_model`.

See **[docs/extending.md](docs/extending.md)** and **[docs/adding_new_models.md](docs/adding_new_models.md)** for details.

### Path Management
**Architecture**: All path generation flows through `AudioLoopConfig`. The `utils/paths.py` module provides internal helpers used by config only.

**Principle**: Production code should never directly call `get_output_dir()` or `get_training_sets_dir()` - always use `AudioLoopConfig` properties and methods.

```python
from audioloop.config import AudioLoopConfig
from audioloop.utils.paths import extract_version_from_filename
from pathlib import Path

# Correct: Use config for all paths
config = AudioLoopConfig(experiment_name="test")
output_dir = config.output_dir           # outputs/test/
training_dir = config.training_sets_dir  # training_sets/test/
model_path = config.get_model_path(1)    # outputs/test/model_v1.pt

# Utility: Version extraction (doesn't build paths)
version = extract_version_from_filename(Path("model_v5.pt"), "model")  # 5
```

**Why this matters**: Centralizing path logic prevents inconsistencies and makes experiment organization reliable across all modules.

### Binary Classification Setup
Any class from supported datasets (UrbanSound8K or FSD50K) can be converted to binary classification by specifying `--class-name` or `--class-id`. The system automatically creates positive/negative labels and appropriate class names.

### Confidence-Based Selection
Active learning prioritizes samples with high model confidence for human review, focusing on potential model errors and boundary cases.

### Separation of Concerns
**DESIGN**: The system maintains clean separation between inference and evaluation:
- **Active Learning Pipeline**: Lean inference engine focused on producing predictions and candidates
- **Metrics Pipeline**: Comprehensive analysis tool for tracking performance across iterations
- **Candidate Selection**: Strategy classes handle pure selection logic, utility functions handle I/O
- **Training Control**: Strategy classes handle stopping decisions, training loop handles execution
- This ensures focused, maintainable code with clear responsibilities

### Dataset Subsetting and Large Dataset Workflows
AudioLoop supports creating training-ready subsets from large datasets, lazy spectrogram generation, and efficient remote deployment with subset-specific spec directories.

See **[docs/custom_datasets.md](docs/custom_datasets.md)** for subsetting, lazy generation, and remote deployment workflows.

### Spectrogram Preprocessing
Audio is converted to variable-length mel-spectrograms with log normalization, stored as PyTorch tensors for efficient loading. Spectrograms can be pre-generated via `create_specs.py` or generated on-demand during training via lazy generation.

### Stopping Criteria
AudioLoop uses pluggable Strategy patterns for both training stopping (within a cycle) and cycle stopping (across cycles):

- **Training stopping**: `PlateauCriterion` (default) stops when loss plateaus, with optional accuracy floor. `AccuracyCriterion` stops at 100% accuracy. See **[docs/stopping_criteria_guide.md](docs/stopping_criteria_guide.md)**.
- **Cycle stopping**: `LabelModeStoppingCriterion` (optimizes F1) and `SearchModeStoppingCriterion` (optimizes recall with precision floor). See **[docs/cycle_stopping_criteria.md](docs/cycle_stopping_criteria.md)**.
- **Custom criteria**: See **[docs/extending.md](docs/extending.md)**.

### Reproducibility and Seed Management
All modules use seed `42` by default. The `set_seed()` function controls Python `random`, NumPy, PyTorch, CUDA, and CUDNN deterministic mode. All CLI scripts accept `--seed` and seeds propagate through automated workflows.

## Related Documentation

- **[docs/custom_datasets.md](docs/custom_datasets.md)**: Adding your own audio data
- **[docs/extending.md](docs/extending.md)**: Adding custom models, strategies, and stopping criteria
- **[docs/adding_new_models.md](docs/adding_new_models.md)**: Detailed model integration guide
- **[docs/candidate_selection_explained.md](docs/candidate_selection_explained.md)**: Deep dive into selection strategies
- **[docs/stopping_criteria_guide.md](docs/stopping_criteria_guide.md)**: Training stopping criteria
- **[docs/cycle_stopping_criteria.md](docs/cycle_stopping_criteria.md)**: Cross-cycle stopping criteria
- **[docs/shape_compatibility_and_variable_lengths.md](docs/shape_compatibility_and_variable_lengths.md)**: Variable-length spectrogram support
- **[webui/README.md](webui/README.md)**: Web-based labeling interface