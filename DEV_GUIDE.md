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
- `AUDIOLOOP_SPECS_DIR`: Spectrograms subdirectory (default: `all_specs`)

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

# No weighting (default) - treats all classes equally
config = AudioLoopConfig(class_weighting=None)

# Adaptive weighting - calculates inverse frequency from training set each cycle
config = AudioLoopConfig(class_weighting="adaptive")

# Fixed weighting - maintains consistent target positive ratio across cycles
config = AudioLoopConfig(class_weighting=0.25)  # Target 25% positive
```

**Implementation**: The `class_weighting` parameter is a unified setting that replaced the previous boolean `use_class_weighting` flag, enabling experimentation with fixed target ratios:
- `None`: Standard CrossEntropyLoss without weights
- `"adaptive"`: Weights calculated as `total_samples / (num_classes * class_counts)` each cycle
- `float` (0.0-1.0): Fixed weights as `weight_positive = (1.0 - target) / target`

**When to Use**:
- **No weighting**: Balanced datasets or as baseline for comparison
- **Adaptive**: Naturally imbalanced data where training set ratio should match
- **Fixed**: When F1 scores oscillate between cycles due to ratio drift (experimental)

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

### Dataset Subsetting
AudioLoop provides a unified interface for creating training-ready subsets from large datasets:

**Purpose**: Large datasets like AudioSet (2M+ files) are impractical to work with in their entirety. The subsetting system creates manageable, labeled subsets for binary classification tasks.

**Key Features**:
- **Unified Interface**: `DatasetConfig.create_subset()` method provides consistent API across datasets
- **Training-Ready Output**: Generates CSV with format: `filename,label,original_class,split,audio_path`
- **Missing File Handling**: Automatically filters out missing files (e.g., deleted YouTube videos)
- **Reproducible Sampling**: Seed-based random sampling for consistent subsets
- **Balanced vs Imbalanced**: Configurable positive ratio for dataset balance

**CLI Tool**:
```bash
# Create subset with default 50% positive ratio
python -m audioloop.create_subset --dataset audioset --class-name "Dog" --max-samples 1000

# Create imbalanced subset (5% positive)
python -m audioloop.create_subset --dataset audioset --class-name "Speech" \
    --max-samples 100000 --positive-ratio 0.05

# List available classes
python -m audioloop.create_subset --dataset audioset --list-classes
```

**Programmatic Usage**:
```python
from audioloop.config import AudioLoopConfig

config = AudioLoopConfig(dataset="audioset")
dataset_config = config.get_dataset_config()

# Create subset
subset_path = dataset_config.create_subset(
    output_path=Path("subsets/dog_1000.csv"),
    class_name="Dog",
    max_samples=1000,
    positive_ratio=0.5,
    split="unbalanced_train",  # Dataset-specific split
    seed=42
)
```

**Output Format**: The generated CSV is self-contained and training-ready:
- `filename`: Audio filename (e.g., "abc123.flac")
- `label`: Binary label (0 or 1)
- `original_class`: Original class name from dataset
- `split`: Dataset split (preserves split info for reproducibility)
- `audio_path`: Full path to audio file (enables lazy spec generation)

### Remote Deployment with Subset Specs

For remote training (cloud pods, HPC clusters), transferring the entire `all_specs` directory is inefficient. AudioLoop provides a tool to create subset-specific spectrogram directories:

**Purpose**: Enable efficient remote deployment by creating self-contained spec directories containing only the files needed for a specific subset.

**Key Features**:
- **Hard Links (Zero Storage)**: Uses hard links by default for local zero-cost duplication
- **Copy Fallback**: Automatically falls back to copying if hard links fail (different filesystems)
- **Self-Contained**: Each subset directory contains exactly the specs needed
- **Simple Syncing**: Standard rsync works without complex --files-from patterns

**Workflow**:
```bash
# 1. Create subset CSV
python -m audioloop.create_subset --dataset audioset --class-name "Dog" --max-samples 100000

# 2. Prepare subset-specific specs directory (hard links, zero storage overhead)
python -m audioloop.prepare_subset_specs subsets/audioset_dog_100000.csv

# 3. Sync to remote (--no-o --no-g avoids permission errors on pods)
rsync -avz --no-o --no-g data/subset_specs/audioset_dog_100000/ user@pod:/workspace/data/specs/

# 4. Train on remote
ssh user@pod "cd /workspace && python -m audioloop.train subset.csv --specs-dir data/specs"
```

**CLI Options**:
```bash
# Custom output directory
python -m audioloop.prepare_subset_specs subsets/audioset_dog_100k.csv \
    --output-dir data/subset_specs/custom_name

# Use copying instead of hard links (different filesystems)
python -m audioloop.prepare_subset_specs subsets/audioset_dog_100k.csv --use-copy

# Custom source specs directory
python -m audioloop.prepare_subset_specs subsets/audioset_dog_100k.csv \
    --specs-dir /path/to/custom_all_specs
```

**Storage Efficiency**: Hard links use zero additional storage locally since they're just directory entries pointing to the same inode. You can delete the subset directory after syncing without affecting `all_specs`.

### Lazy Spectrogram Generation
AudioLoop supports on-demand spectrogram generation during training:

**How It Works**:
1. **CSV with audio_path**: Training CSVs can include full path to audio files
2. **Dataset Config Pass-Through**: `SpectrogramDataset` receives `dataset_config` parameter
3. **On-Demand Generation**: When `.pt` file missing, generates spec from audio on-the-fly
4. **Automatic Caching**: Generated specs saved to disk for future use

**Benefits**:
- **Simpler Workflow**: Fewer pre-processing steps
- **Storage Efficiency**: Only generate specs for files actually used
- **Flexibility**: Change spectrogram parameters without regenerating entire dataset

**Implementation**:
```python
# In training_core.py
dataset_config = config.get_dataset_config()

train_dataset = SpectrogramDataset(
    csv_file=labels_file,
    specs_dir=str(config.specs_dir),
    dataset_config=dataset_config,  # Enables lazy generation
)
```

**Behavior**:
- Spectrogram `.pt` file exists → Load from disk (fast path)
- Spectrogram missing + `audio_path` available + `dataset_config` provided → Generate on-the-fly
- Spectrogram missing + no lazy generation support → Raise helpful error

**Example Workflows**:
```bash
# Approach 1: Pre-generate all specs
python -m audioloop.create_specs --dataset audioset
python -m audioloop.train training_set_v1.csv

# Approach 2: Lazy generation
python -m audioloop.create_subset --dataset audioset --class-name "Dog" --max-samples 1000
python -m audioloop.train subsets/audioset_dog_1000.csv  # Specs generated as needed
```

### Spectrogram Preprocessing
Audio is converted to variable-length mel-spectrograms with log normalization, stored as PyTorch tensors for efficient loading. Spectrograms can be pre-generated via `create_specs.py` or generated on-demand during training via lazy generation.

### Pluggable Training Stopping Criteria
AudioLoop uses a Strategy pattern for training stopping decisions:
- **Architecture**: Abstract base class `TrainingStoppingCriterion` with pluggable implementations
- **Default Behavior**: `HybridStoppingCriterion` adaptively switches between accuracy and plateau modes based on training behavior
- **Strategy Classes**: `AccuracyCriterion`, `PlateauCriterion`, `HybridStoppingCriterion`
- **Extensibility**: Easy to add early stopping, plateau detection, or custom criteria
- **Interface**: `should_stop(epoch, train_accuracy, train_loss, val_accuracy, val_loss) -> bool`
- **State Management**: `reset()` method for stateful criteria like early stopping
- **Usage**: Pass `stopping_criterion` parameter to `run_training()` or use default

### Cycle Stopping Criteria
AudioLoop also supports automatic stopping of active learning cycles based on candidate performance metrics:
- **Architecture**: Base class `CycleStoppingCriterion` with mode-specific implementations
- **Strategies**: `LabelModeStoppingCriterion` (optimizes F1), `SearchModeStoppingCriterion` (optimizes recall with precision floor)
- **Metrics Tracking**: Candidate metrics calculated after each labeling round and persisted to JSON
- **Rolling Statistics**: Uses rolling averages and stability measures to reduce noise from small sample sizes
- **State Management**: Tracks patience counter and best cycle across the active learning process
- **Interface**: `should_stop(current_cycle) -> bool` and `get_best_cycle() -> int`
- **Integration**: `automated_workflow.py` and `merge_labels.py` handle metric calculation and stopping decisions
- **Factory**: `create_cycle_stopping_criterion(config, metrics_history)` for instantiation
- **Limitations**: Experimental feature - candidate metrics (on ~50 high-entropy examples) don't always correlate with corpus performance

See `docs/cycle_stopping_criteria.md` for detailed documentation.

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