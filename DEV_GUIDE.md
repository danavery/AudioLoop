# DEV_GUIDE.md

## Project Overview

AudioLoop is an active learning framework for binary audio classification supporting arbitrary audio datasets. It implements a versioned workflow for iterative model improvement through human-in-the-loop labeling. Built-in support includes FSD50K and UrbanSound8K, with easy extensibility for custom datasets.

## Project Goal

The eventual goal is to enable a user to arrive at the application with only a large unlabeled audio dataset and a small labeled subset of that dataset, and let the system independently create a model that can label the entire dataset with high accuracy. That means potentially automatically setting hyperparameters for the entire training process including early stopping strategies, learning rate scheduling, candidate selection strategies for the human labeling set.
That includes both CLI and web interfaces for the training loop and the human labeling UI.

## Backwards Compatibility

This is a project in rapid development and has no production users. Maintaining backwards compatibility during code changes is not required and will only complicate the codebase unnecessarily. Do not be concerned about backwards compatibility.

## Workflow Modes

AudioLoop supports two primary workflow modes designed for different use cases:

### Production Mode (Default)
**Use case**: Real-world deployment with truly unlabeled datasets
- **Ground truth**: Not available - you don't know the true labels
- **Active learning**: Generates predictions without ground truth columns
- **Metrics tracking**: Shows prediction and confidence metrics only
- **Human labeling**: Manual review using web UI or terminal interface
- **Goal**: Build models for actual unknown audio classification tasks

### Evaluation Mode (Research/Testing)
**Use case**: Research, development, and algorithm testing with known datasets
- **Ground truth**: Available - you have the true labels for comparison
- **Active learning**: Use `--with-ground-truth` flag to include evaluation columns
- **Metrics tracking**: Shows full evaluation metrics (F1, precision, recall, accuracy)
- **Auto-labeling**: Can use `auto_label_candidates.py` for rapid testing
- **Goal**: Test candidate selection strategies, tune hyperparameters, compare methods

**Key Distinction**: Production mode is for real deployment where ground truth is unknown. Evaluation mode is for research and testing where ground truth is available for validation.

## Common Commands

### Data Preparation (One-time Setup)
```bash
# Generate spectrograms for entire FSD50K dataset
python -m audioloop.create_all_specs
```

### Creating Initial Training Sets
```bash
# Create FSD50K training set (default)
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50

# Create UrbanSound8K training set
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --class-name siren --n 40

# Create with custom parameters
python -m audioloop.utils.create_bootstrap_set --class-name Speech --n 60 --positive-pct 0.8 --output training_sets/training_set_v2.csv

# Create with custom seed for reproducible sampling
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50 --seed 123

# List available classes for FSD50K
python -m audioloop.utils.create_bootstrap_set --list-classes

# List available classes for UrbanSound8K
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --list-classes
```

### Training Models
```bash
# Train model (version auto-detected from filename)
python -m audioloop.train training_sets/training_set_v1.csv

# Train with explicit version and parameters
python -m audioloop.train training_sets/training_set_v1.csv -v 1 --epochs 500 --batch-size 64

# Train with experiment name (outputs to outputs/myexp/)
python -m audioloop.train training_sets/training_set_v1.csv --experiment myexp

# Train with custom seed for reproducibility
python -m audioloop.train training_sets/training_set_v1.csv --seed 123

# Train with specific stopping criterion
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion hybrid
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion accuracy
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion plateau --patience 30

# Custom hybrid stopping parameters
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion hybrid \
  --high-accuracy-threshold 0.9 --high-accuracy-patience 15 --patience 25
```

### Active Learning Workflow
```bash
# Run active learning cycle (auto-detects version from run number)
python -m audioloop.active_learning --class-name Drill --run-number 1

# Run with UrbanSound8K dataset
python -m audioloop.active_learning --dataset urbansound8k --class-name siren --run-number 1

# Run with entropy-based selection
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode entropy

# Run with custom seed for reproducible candidate selection
python -m audioloop.active_learning --class-name Drill --run-number 1 --seed 123

# Run with explicit parameters
python -m audioloop.active_learning --class-name Speech --run-number 2 --total-candidates 20 --positive-pct 0.75 --min-confidence 0.85

# Run with experiment name (uses outputs/myexp/ instead of outputs/)
python -m audioloop.active_learning --class-name Drill --run-number 1 --experiment myexp

# Run with entropy-based selection (uncertainty sampling)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode entropy

# Run with basic transition (starts with confidence, switches to entropy when criteria met)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition

# Run with auto-calculated thresholds (recommended for imbalanced datasets)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition --auto-thresholds --estimated-positive-pct 0.10

# Run with auto-thresholds using default estimate (5%)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition --auto-thresholds

# Run with custom basic transition thresholds (manual tuning)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition \
  --basic-transition-f1-threshold 0.25 --basic-transition-confidence-threshold 0.95 --basic-transition-variance-threshold 0.10

# List available sound classes (respects AUDIOLOOP_DATASET environment variable)
python -m audioloop.active_learning --list-classes

# List UrbanSound8K classes
python -m audioloop.active_learning --dataset urbansound8k --list-classes

# Include ground truth evaluation columns (for research/evaluation with labeled datasets)
python -m audioloop.active_learning --class-name Drill --run-number 1 --with-ground-truth

# Production workflow (default - no ground truth columns, for real unlabeled datasets)
python -m audioloop.active_learning --class-name Drill --run-number 1
```

### Human Labeling

#### Web UI (Recommended)
```bash
# Install web UI dependencies
uv sync --extra webui

# Start web labeling interface
cd webui && python app.py

# Open browser to http://127.0.0.1:5000
# Load: outputs/labeling_candidates_v1.csv
# Features: Visual interface, audio player, keyboard shortcuts, progress tracking
```

#### Terminal Interface (Alternative)
```bash
# Interactive audio labeling tool (uses AUDIOLOOP_DATASET or default)
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# Explicit FSD50K dataset
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset fsd50k --audio-dir data/FSD50K/FSD50K.dev_audio

# UrbanSound8K with environment variable
AUDIOLOOP_DATASET=urbansound8k python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# Explicit UrbanSound8K dataset
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset urbansound8k --audio-dir data/urbansound8k

# For auto-labeling with ground truth (evaluation workflows), see auto_label_candidates section below
```

### Label Management

#### Production Workflow (Manual Labeling)
```bash
# Merge human labels back into training set (default experiment)
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Merge labels with explicit experiment name
python -m audioloop.merge_labels training_sets/myexp/training_set_v1.csv outputs/myexp/labeling_candidates_v1.csv --experiment myexp

# Merge with auto-generated output path
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv --experiment myexp
```

#### Evaluation Workflow (Auto-Labeling for Research)
```bash
# Auto-label candidates using ground truth (requires --with-ground-truth mode)
python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv

# Auto-label with verbose progress output
python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv --verbose

# Auto-label candidates from experiment
python -m audioloop.auto_label_candidates outputs/myexp/labeling_candidates_v2.csv

# Note: Auto-labeling requires ground truth data in CSV (generated with --with-ground-truth)
# For production workflows without ground truth, use manual labeling tools instead
```

### Comprehensive Metrics Tracking
```bash
# Track metrics (automatically detects evaluation vs production mode)
python -m audioloop.track_metrics

# Generate metrics plots - shows different visualizations based on available data
python -m audioloop.track_metrics --plot

# Save metrics plots to file
python -m audioloop.track_metrics --save-plot comprehensive_metrics.png

# Track metrics from experiment directory
python -m audioloop.track_metrics --experiment myexp --plot

# Evaluation mode: Shows F1, precision, recall, accuracy (when ground truth available)
# Production mode: Shows prediction and confidence metrics only (no ground truth)
# Mixed mode: Handles files with and without ground truth gracefully
```

### Outputs Management
```bash
# Analyze what files can be cleaned (dry run)
python -m audioloop.clean_outputs

# Clean safe files (removes example/demo files, keeps workflow files)
python -m audioloop.clean_outputs --clean

# Clean without confirmation
python -m audioloop.clean_outputs --clean --force

# Move misplaced training files to training_sets/
python -m audioloop.clean_outputs --move-training
```

### Code Quality
```bash
# Format and lint code
ruff check audioloop/
ruff format audioloop/
```

### Automated Workflow (Recommended)
```bash
# Evaluation workflow (auto-labeling for testing/development)
python -m audioloop.automated_workflow --class-name Drill --cycles 3 --evaluation-mode --auto-label

# Production workflow (pause for human labeling)
python -m audioloop.automated_workflow --class-name Speech --cycles 2

# Custom training parameters
python -m audioloop.automated_workflow --class-name Music --cycles 3 --epochs 500 --batch-size 64

# Reproducible automated workflow with custom seed
python -m audioloop.automated_workflow --class-name Drill --cycles 3 --evaluation-mode --auto-label --seed 123

# Custom active learning parameters
python -m audioloop.automated_workflow --class-name Explosion --cycles 2 --candidates 100 --positive-pct 0.8

# UrbanSound8K dataset (evaluation mode)
python -m audioloop.automated_workflow --class-name siren --cycles 2 --dataset urbansound8k --evaluation-mode --auto-label

# Use entropy-based selection
python -m audioloop.automated_workflow --class-name Drill --cycles 3 --selection-mode entropy

# Use basic transition (automatically switches strategies based on performance)
python -m audioloop.automated_workflow --class-name Drill --cycles 3 --selection-mode basic_transition --evaluation-mode --auto-label

# Mix confidence and entropy modes across cycles
python -m audioloop.automated_workflow --class-name Speech --cycles 4 --selection-mode confidence --evaluation-mode --auto-label

# Run experiment with custom output directory
python -m audioloop.automated_workflow --class-name siren --cycles 3 --evaluation-mode --auto-label --experiment myexp
```

### Complete Workflow Examples

#### Production Workflow (Real Deployment)
**Use case**: Deploy on unknown audio data for actual classification tasks

```bash
# 1. Create initial training set for experiment
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40 --experiment myexp

# 2. Train initial model
python -m audioloop.train training_sets_myexp/training_set_v1.csv --experiment myexp

# 3. Run active learning cycle (production mode - no ground truth)
python -m audioloop.active_learning --class-name siren --run-number 1 --experiment myexp

# 4. Label candidates manually using web UI (recommended)
cd webui && python app.py
# Load: outputs/myexp/labeling_candidates_v1.csv

# Alternative: Terminal interface
python -m audioloop.label_audio outputs/myexp/labeling_candidates_v1.csv

# 5. Merge human labels (creates training_sets/myexp/training_set_v2.csv)
python -m audioloop.merge_labels training_sets/myexp/training_set_v1.csv outputs/myexp/labeling_candidates_v1.csv

# 6. Track metrics (production mode - confidence and prediction metrics only)
python -m audioloop.track_metrics --experiment myexp --plot
```

#### Evaluation Workflow (Research/Testing)
**Use case**: Test algorithms, compare strategies, tune parameters with known datasets

```bash
# 1. Create initial training set for experiment
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40 --experiment eval_test

# 2. Train initial model
python -m audioloop.train training_sets_eval_test/training_set_v1.csv --experiment eval_test

# 3. Run active learning cycle WITH ground truth evaluation
python -m audioloop.active_learning --class-name siren --run-number 1 --experiment eval_test --with-ground-truth

# 4. Auto-label candidates using ground truth (for rapid testing)
python -m audioloop.auto_label_candidates outputs/eval_test/labeling_candidates_v1.csv

# 5. Merge labels (creates training_sets_eval_test/training_set_v2.csv)
python -m audioloop.merge_labels training_sets/eval_test/training_set_v1.csv outputs/eval_test/labeling_candidates_v1.csv --experiment eval_test

# 6. Track comprehensive metrics (evaluation mode - F1, precision, recall, accuracy)
python -m audioloop.track_metrics --experiment eval_test --plot
```

**Key Differences**:
- **Production**: No `--with-ground-truth`, manual labeling, confidence metrics only
- **Evaluation**: Uses `--with-ground-truth`, auto-labeling possible, full evaluation metrics

**Use automated workflow for**: Convenience, testing multiple cycles quickly, parameter sweeping  
**Use manual workflow for**: Learning the system, debugging issues, fine-grained control

## Selection Strategies

AudioLoop uses a pluggable strategy pattern for candidate selection in active learning:

### ConfidenceStrategy (Default)
- **Class**: `ConfidenceStrategy`
- **Algorithm**: Selects samples with highest model confidence scores
- **Use Case**: Early training cycles when model is uncertain
- **Behavior**: Focuses on samples the model is most sure about
- **Risk**: Can lead to overconfidence and performance degradation in later cycles
- **Selection Mode**: `--selection-mode confidence`

### EntropyStrategy (Uncertainty Sampling)
- **Class**: `EntropyStrategy`
- **Algorithm**: Selects samples with highest entropy (most uncertain predictions)
- **Use Case**: Later training cycles or when model becomes overconfident
- **Behavior**: Focuses on samples near decision boundaries
- **Benefit**: Helps model learn challenging cases and avoid overconfidence
- **Selection Mode**: `--selection-mode entropy`

### Strategy Architecture
Each strategy class implements the `CandidateSelectionStrategy` interface:
```python
from audioloop.utils.candidate_selection import ConfidenceStrategy, EntropyStrategy

# Direct instantiation (like stopping criteria)
strategy = ConfidenceStrategy()
candidates = strategy.select_candidates(predictions, num_candidates=50)
```

### Recommended Usage
1. **Start with ConfidenceStrategy** for initial cycles to establish basic patterns
2. **Switch to EntropyStrategy** when model becomes overconfident (mean confidence >0.95, std confidence <0.11)
3. **Monitor F1 score** - switch strategies if performance starts degrading

### Output Format
The active learning script displays strategy class names in the output:
```
Running active learning cycle
------------------------------------------------------------
Selection strategy: ConfidenceStrategy
------------------------------------------------------------

Step 2: Selecting candidates for human labeling...
Using strategy: ConfidenceStrategy
```

### Example Workflow
```bash
# Complete workflow demonstration with basic transition (evaluation mode)
python -m audioloop.automated_workflow --class-name Drill --cycles 2 --selection-mode basic_transition --evaluation-mode --auto-label

# Manual workflow demonstration (production mode)
python -m audioloop.automated_workflow --class-name Drill --cycles 2
```

### Basic Transition Configuration
```bash
# Default basic transition thresholds
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition

# Custom thresholds for sensitive datasets
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition \
  --basic-transition-f1-threshold 0.15 --basic-transition-confidence-threshold 0.85 --basic-transition-variance-threshold 0.15

# Stricter thresholds for high-confidence models
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition \
  --basic-transition-f1-threshold 0.3 --basic-transition-confidence-threshold 0.95 --basic-transition-variance-threshold 0.08
```

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

**📖 For adding new models, see [Adding New Models Guide](docs/adding_new_models.md)**

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

### Development Patterns

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
