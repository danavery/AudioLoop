# AudioLoop Developer Guide

Developer reference for AudioLoop architecture, patterns, and extensibility. For usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md). For workflow patterns, see [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md).

## Project Architecture

AudioLoop is an active learning framework for binary audio classification supporting arbitrary audio datasets. It implements a versioned workflow for iterative model improvement through human-in-the-loop labeling. Built-in support includes FSD50K and UrbanSound8K, with easy extensibility for custom datasets.

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

Use the `--with-ground-truth` flag to enable evaluation mode in active learning commands.

### Selection Strategies

AudioLoop uses a pluggable strategy pattern for candidate selection:

**ConfidenceStrategy** (Default):
- Selects samples with highest model confidence scores
- Best for early training cycles when model is uncertain
- Can lead to overconfidence in later cycles

**EntropyStrategy** (Uncertainty Sampling):
- Selects samples with highest entropy (most uncertain predictions)
- Best for later training cycles or when model becomes overconfident
- Focuses on samples near decision boundaries

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
- **`create_all_specs.py`**: Preprocesses audio into spectrograms
- **`track_metrics.py`**: Comprehensive metrics tracking and visualization (accuracy, F1, precision, recall, confidence, entropy) across active learning cycles
- **`config.py`**: Unified configuration system coordinating paths, datasets, and experiments
- **`utils/create_bootstrap_set.py`**: Bootstrap training set creation from ground truth (evaluation mode only)
- **`utils/paths.py`**: Path utilities eliminating hardcoded path duplication
- **`utils/stopping_criteria.py`**: Pluggable training stopping criteria using Strategy pattern
- **`utils/candidate_selection.py`**: Pluggable candidate selection strategies using Strategy pattern

### Models
- **`models/base.py`**: Abstract base class (`AudioLoopModel`) defining the pluggable model interface
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

# Create configuration
config = AudioLoopConfig(experiment_name="my_exp", dataset="urbansound8k")

# Access all path locations
config.output_dir          # outputs/my_exp/
config.training_sets_dir   # training_sets/my_exp/
config.specs_dir          # data/all_specs/

# Generate versioned file paths
config.get_model_path(1)        # outputs/my_exp/model_v1.pt
config.get_predictions_path(1)  # outputs/my_exp/predictions_v1.csv
config.get_training_set_path(1) # training_sets_my_exp/training_set_v1.csv
```

### Configuration Precedence
AudioLoop follows standard configuration patterns with proper precedence:
1. **Explicit constructor parameters** (highest priority)
2. **Environment variables** (fallback when no explicit value)
3. **Default values** (lowest priority)

```python
# Environment variable as fallback
os.environ['AUDIOLOOP_DATASET'] = 'urbansound8k'
config = AudioLoopConfig()  # Uses urbansound8k

# Explicit parameter overrides environment
config = AudioLoopConfig(dataset='fsd50k')  # Uses fsd50k (ignores env var)
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
- `AUDIOLOOP_SPECS_DIR`: Spectrograms subdirectory (default: `all_specs`)

### Sound Classification
- **`datasets/fsd50k.py`**: FSD50K class mappings (200 classes with semantic groupings)
- **`datasets/urbansound8k.py`**: UrbanSound8K class mappings (10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music)
- Binary classification framework converts any class into positive/negative labels

## Data Flow

### Production Mode (Default)
1. **Preprocessing**: Raw audio → spectrograms via `create_all_specs.py`
2. **Initial Training Set**: User-provided labeled dataset
3. **Model Training**: Small labeled set → CNN model via `train.py`
4. **Active Learning**: Model predictions on ALL files → candidate selection via `active_learning.py`
5. **Human Labeling**: Audio playback + labeling via `label_audio.py` or web UI
6. **Label Integration**: Human labels → expanded training set via `merge_labels.py`
7. **Performance Monitoring**: Prediction and confidence metrics via `track_metrics.py`
8. **Iteration**: Repeat training with expanded data

### Evaluation Mode (Research/Testing)
1. **Preprocessing**: Raw audio → spectrograms via `create_all_specs.py` 
2. **Bootstrap Training Set**: Sample from ground truth via `utils/create_bootstrap_set.py`
3. **Model Training**: Small labeled set → CNN model via `train.py`
4. **Active Learning**: Model predictions with ground truth → candidate selection via `active_learning.py --with-ground-truth`
5. **Auto-Labeling**: Ground truth extraction via `auto_label_candidates.py` (optional)
6. **Label Integration**: Labels → expanded training set via `merge_labels.py`
7. **Performance Evaluation**: Full ground truth metrics (F1, precision, recall) via `track_metrics.py`
8. **Iteration**: Repeat training with expanded data

**Key Difference**: Evaluation mode includes ground truth data for comprehensive performance analysis, while production mode works with truly unknown data.

### Clean Architecture
- **Active Learning**: Focused inference engine that produces predictions and candidate selections
- **Metrics Tracking**: Comprehensive performance analysis across active learning iterations
- **Candidate Selection**: Uses only model outputs (confidence, entropy) for selection decisions
- **Strategy Pattern**: Consistent pluggable architecture for both training stopping criteria and candidate selection strategies

## File Formats

### Training Set CSV
```csv
filepath,label,run
data/all_specs/100032-3-0-0.pt,1,1
data/all_specs/100263-2-0-117.pt,0,1
```

### Predictions CSV (Generated)
Default format (production mode):
```csv
filename,prediction,predicted_class,confidence,entropy,prob_negative,prob_positive,original_class,fold,filepath
```

With ground truth evaluation (--with-ground-truth flag):
```csv
filename,ground_truth,prediction,predicted_class,confidence,entropy,prob_negative,prob_positive,correct,original_class,fold,filepath
```

### Candidates CSV (For Human Labeling)
```csv
filename,prediction,predicted_class,confidence,needs_human_label,entropy,prob_negative,prob_positive,original_class,fold,filepath
```

## Key Dependencies

- **PyTorch**: Neural network training and inference
- **TorchAudio**: Audio processing and spectrogram generation
- **NumPy**: Numerical operations
- **SoundFile**: Audio file I/O
- **TQDM**: Progress bars
- **Ruff**: Code formatting and linting

## Development Patterns

### Unified Configuration Pattern
All modules use the unified configuration system instead of scattered parameters:
```python
# Modern approach - unified configuration
from audioloop.config import AudioLoopConfig
config = AudioLoopConfig(experiment_name="test", dataset="urbansound8k")

# Pass config object to functions
run_training(config, labels_file="training.csv", ...)
run_active_learning_cycle(config, positive_class_name="Drill", ...)
```

### Strategy Pattern Implementation
Both training stopping criteria and candidate selection follow the same Strategy pattern:
- Abstract base class defines the interface
- Concrete strategy classes implement specific algorithms
- Direct class imports and instantiation (no factory functions)
- Each strategy has complete control over its domain logic

Example usage:
```python
# Training stopping criteria
from audioloop.utils.stopping_criteria import AccuracyCriterion
criterion = AccuracyCriterion(max_epochs=1000)

# Candidate selection strategies
from audioloop.utils.candidate_selection import ConfidenceStrategy, EntropyStrategy
strategy = ConfidenceStrategy()
```

### Dataset Extensibility
AudioLoop supports adding custom datasets through a simple file-based convention with automatic discovery:

#### Quick Start
1. **Create config file**: `audioloop/datasets/my_dataset_config.py`
2. **Define config class**: `class MyDatasetConfig(DatasetConfig)`
3. **Implement required methods**: Inherit from `DatasetConfig` and implement abstract methods
4. **Use immediately**: `--dataset my_dataset` in all CLI commands

#### Example - Using the Template
```bash
# 1. Copy the template to create your dataset config
cp audioloop/datasets/templates/simple_audio_template.py audioloop/datasets/my_dataset_config.py

# 2. Edit the copied file:
#    - Rename class from TemplateAudioConfig to MyDatasetConfig
#    - Update paths: data/YOUR_DATASET_NAME/ → data/my_dataset/
#    - Customize class vocabulary for your classes

# 3. Use immediately
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-classes
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-splits
```

Your CSV format:
```csv
filename,label
audio1.wav,speech
audio2.wav,music
audio3.wav,noise
```

#### Naming Convention
- **File naming**: `{dataset_name}_config.py` → dataset name `"{dataset_name}"`
- **Class naming**: `{DatasetName}Config` (e.g., `MyAudioConfig`, `CommonVoiceConfig`)
- **Auto-discovery**: No registration needed - just create the file

#### Available Datasets
List currently available datasets:
```bash
# Method 1: Check help message (shows available datasets)
python -m audioloop.utils.create_bootstrap_set --help

# Method 2: Check via Python
python -c "from audioloop.datasets.registry import list_available_datasets; print(list_available_datasets())"
```

### Model Extensibility
AudioLoop supports adding custom models through a simple file-based convention with automatic discovery:

#### Quick Start
1. **Create model file**: `audioloop/models/my_model.py`
2. **Define model class**: Standard PyTorch `nn.Module` that inherits from `AudioLoopModel`
3. **Implement minimal interface**: Just one method - `get_model_info()`
4. **Use immediately**: `--model-type my_model` in training commands

#### Requirements
All custom models must:
- **Inherit from `AudioLoopModel`**: Minimal abstract base class (just metadata)
- **Accept standard parameters**: `num_classes` and `**kwargs` in constructor
- **Use standard PyTorch interface**: Standard `forward(x: torch.Tensor)` method
- **Return logits**: Standard classification output format

#### Example Implementation
```python
# audioloop/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .audio_loop_model import AudioLoopModel

class ResNet(AudioLoopModel):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depth = kwargs.get('depth', 18)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)

        # Build ResNet architecture based on depth
        # ... implement ResNet layers ...
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        # ... more layers ...
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        # Add channel dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Standard ResNet forward pass
        x = F.relu(self.conv1(x))
        # ... forward through ResNet blocks ...

        # Global pooling and classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_model_info(self) -> dict:
        """Get model metadata - constructor parameters are auto-saved."""
        return {
            "model_type": "resnet",  # Should match filename
            "num_classes": self.num_classes,
            "depth": self.depth,
            "dropout_rate": self.dropout_rate,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
```

#### Usage
```bash
# List available models
python -m audioloop.train --list-models

# Train with custom model (basic usage)
python -m audioloop.train training_set_v1.csv --model-type resnet

# Models automatically appear in all CLI commands
python -m audioloop.active_learning --class-name Drill --run-number 1  # Uses saved model
```

#### Custom Model Parameters
Pass model-specific parameters via config:
```python
from audioloop.config import AudioLoopConfig

# Configure custom model parameters
config = AudioLoopConfig(
    model_type="resnet",
    model_kwargs={
        "depth": 50,
        "dropout_rate": 0.2,
        "custom_param": "value"
    }
)

# Use in training - parameters are automatically preserved
from audioloop.training_core import run_training
run_training(config, labels_file="training_set_v1.csv", version=1)
```

**Note**: AudioLoop automatically saves and restores all constructor parameters from `get_model_info()`, so models are reconstructed with exactly the same configuration.

#### Key Features
- **Standard PyTorch**: Uses normal `forward(x: torch.Tensor)` - full ecosystem compatibility
- **Automatic Save/Load**: Constructor parameters preserved automatically
- **File naming**: `my_model.py` → model name `"my_model"`
- **Class naming**: Any name - system finds the AudioLoopModel subclass automatically
- **Auto-discovery**: No registration needed - just create the file

#### Available Models
List currently available models:
```bash
# Method 1: Check training help
python -m audioloop.train --list-models

# Method 2: Check via Python
python -c "from audioloop.models.model_registry import list_available_models; print(list_available_models())"
```

### Path Management
Centralized path utilities eliminate hardcoded path duplication:
```python
from audioloop.utils.paths import get_output_dir, get_specs_dir, extract_version_from_filename

# Environment-configurable paths
output_dir = get_output_dir("experiment_name")  # outputs/experiment_name/
specs_dir = get_specs_dir()                     # data/all_specs/ (or custom)

# Version extraction
version = extract_version_from_filename(Path("model_v5.pt"), "model")  # 5
```

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

### Spectrogram Preprocessing
Audio is converted to fixed-length (993 frames) mel-spectrograms with log normalization, stored as PyTorch tensors for efficient loading.

### Pluggable Training Stopping Criteria
AudioLoop uses a Strategy pattern for training stopping decisions:
- **Architecture**: Abstract base class `TrainingStoppingCriterion` with pluggable implementations
- **Default Behavior**: `HybridStoppingCriterion` adaptively switches between accuracy and plateau modes based on training behavior
- **Strategy Classes**: `AccuracyCriterion`, `PlateauCriterion`, `HybridStoppingCriterion`
- **Extensibility**: Easy to add early stopping, plateau detection, or custom criteria
- **Interface**: `should_stop(epoch, train_accuracy, train_loss, val_accuracy, val_loss) -> bool`
- **State Management**: `reset()` method for stateful criteria like early stopping
- **Usage**: Pass `stopping_criterion` parameter to `run_training()` or use default

### Reproducibility and Seed Management
AudioLoop provides seed management for reproducible experiments:

- **Default Seed**: All modules use seed `42` by default for consistent behavior
- **Training Reproducibility**: Model weights, data shuffling, and optimization are seeded
- **Candidate Selection Reproducibility**: Random sampling in active learning is seeded
- **Initial Dataset Creation**: Training set sampling is seeded
- **CLI Support**: All main scripts accept `--seed` parameter for custom seeds
- **Automated Workflows**: Seed propagates through entire multi-cycle workflows

The `set_seed()` function controls:
- Python's `random` module (sampling, shuffling)
- NumPy RNG (numerical operations)
- PyTorch RNG (model initialization, dropout)
- PyTorch CUDA RNG (GPU operations)
- CUDNN deterministic mode (consistent GPU behavior)

### Adaptive Hybrid Stopping Strategy (Default)
The `HybridStoppingCriterion` addresses the challenge where training can get stuck at high accuracy (95-99%) without reaching perfect accuracy:
- **Starts with accuracy-based stopping**: Waits for 100% accuracy to handle erratic early training
- **Monitors high accuracy plateau**: If accuracy ≥ 95% for multiple epochs without reaching 100%
- **Switches to plateau-based stopping**: Efficiently stops when stuck at high accuracy
- **Behavior-driven switching**: Based on actual training dynamics, not arbitrary thresholds

Example usage:
```python
from audioloop.utils.stopping_criteria import HybridStoppingCriterion

# Default configuration (recommended for most audio tasks)
criterion = HybridStoppingCriterion()

# Custom configuration for different training behaviors
criterion = HybridStoppingCriterion(
    high_accuracy_threshold=0.9,    # Switch threshold (90% vs 95%)
    high_accuracy_patience=10,      # Epochs to wait at high accuracy
    plateau_patience=30,            # Plateau detection patience
    max_epochs=1000
)
```

### Available Stopping Criteria
- **`HybridStoppingCriterion`** (default): Adaptive switching from accuracy to plateau mode
- **`AccuracyCriterion`**: Stops at 100% accuracy or max epochs
- **`PlateauCriterion`**: Stops when loss plateaus (early stopping)

## Related Documentation

For specific guides, see:
- **[docs/adding_new_models.md](docs/adding_new_models.md)**: Detailed guide for integrating custom models
- **[docs/candidate_selection_explained.md](docs/candidate_selection_explained.md)**: Deep dive into selection strategies
- **[docs/stopping_criteria_guide.md](docs/stopping_criteria_guide.md)**: Training stopping criteria
- **[docs/shape_compatibility_and_variable_lengths.md](docs/shape_compatibility_and_variable_lengths.md)**: Variable-length spectrogram support
- **[FSD50K_INTEGRATION.md](FSD50K_INTEGRATION.md)**: FSD50K dataset integration
- **[LABELING_GUIDE.md](LABELING_GUIDE.md)**: Audio labeling best practices
- **[webui/README.md](webui/README.md)**: Web-based labeling interface