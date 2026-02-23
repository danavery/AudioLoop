# AudioLoop Usage Guide

Complete command reference for AudioLoop's CLI tools and practical usage patterns. This guide covers all commands, parameters, and examples in one place.

## Installation and Setup

AudioLoop is installed once in a fixed location, then used from separate project directories.

### Install AudioLoop (One-Time)

```bash
# Clone to a permanent location
git clone <repository> ~/tools/audioloop
cd ~/tools/audioloop
uv sync

# Recommended: Add activation alias to ~/.bashrc or ~/.zshrc
echo 'alias al="source ~/tools/audioloop/.venv/bin/activate"' >> ~/.bashrc
```

### Create a Project Directory (Per Classification Task)

```bash
# Activate audioloop
source ~/tools/audioloop/.venv/bin/activate  # Or: al

# Create and initialize project
mkdir ~/projects/my-classifier
cd ~/projects/my-classifier
python -m audioloop.init_project
```

### Environment Variables (Optional)

Set defaults in your shell or project's `audioloop.yaml`:

```bash
export AUDIOLOOP_DATASET=urbansound8k      # Default dataset
export AUDIOLOOP_DATA_ROOT=/custom/data    # Custom data directory
export AUDIOLOOP_OUTPUT_ROOT=/custom/out   # Custom output directory
```

## Configuration System

AudioLoop uses a unified configuration system with proper precedence:

**Configuration Precedence:**
1. **Explicit CLI parameters** (highest priority)
2. **Environment variables** (fallback when no explicit value)
3. **Default values** (lowest priority)

**Environment Variables:**
```bash
export AUDIOLOOP_DATASET=urbansound8k      # Default dataset
export AUDIOLOOP_DATA_ROOT=/custom/data    # Custom data directory  
export AUDIOLOOP_OUTPUT_ROOT=/custom/out   # Custom output directory
export AUDIOLOOP_SPECS_DIR=all_specs       # Spectrograms subdirectory
```

**Experiment Organization:**
AudioLoop automatically organizes files by experiment:
- **Default**: `outputs/`, `training_sets/`
- **With experiment**: `outputs/{experiment}/`, `training_sets/{experiment}/`

## Data Preparation

### Create Dataset Subsets (For Large Datasets)

For large datasets like AudioSet (2M+ files), create manageable subsets for binary classification tasks:

```bash
# Create subset with default 50% positive ratio
python -m audioloop.create_subset --dataset audioset --class-name "Dog" --max-samples 1000

# Create imbalanced subset (5% positive for rare classes)
python -m audioloop.create_subset --dataset audioset --class-name "Speech" \
    --max-samples 100000 --positive-ratio 0.05

# Specify dataset split
python -m audioloop.create_subset --dataset audioset --class-name "Music" \
    --max-samples 50000 --split unbalanced_train

# Custom output location
python -m audioloop.create_subset --dataset audioset --class-name "Dog" \
    --max-samples 1000 --output subsets/custom_dog.csv

# Reproducible subset with custom seed
python -m audioloop.create_subset --dataset audioset --class-name "Dog" \
    --max-samples 1000 --seed 123

# List available classes
python -m audioloop.create_subset --dataset audioset --list-classes
```

**Subset Parameters:**
```bash
--dataset {audioset}           # Dataset to create subset from
--class-name CLASS_NAME        # Class name for positive samples (required)
--max-samples N                # Maximum total samples (positive + negative)
--positive-ratio 0.5           # Target ratio of positive samples (0.0-1.0)
--split SPLIT                  # Dataset split to sample from (default: dataset-specific)
--output OUTPUT_PATH           # Output CSV path (default: subsets/DATASET_CLASS_SAMPLES.csv)
--seed 42                      # Random seed for reproducibility
--list-classes                 # List available classes and exit
```

**Output Format**: Generated CSV includes:
- `filename`: Audio filename
- `label`: Binary label (0 or 1)
- `original_class`: Original class name from dataset
- `split`: Dataset split (for reproducibility)
- `audio_path`: Full path to audio file (enables lazy spec generation)

**Notes:**
- Automatically filters out missing files (e.g., deleted YouTube videos)
- Creates `subsets/` directory if it doesn't exist
- CSV is training-ready and supports lazy spectrogram generation

### Generate Spectrograms (Optional)

Spectrograms can be pre-generated or generated on-demand during training. Pre-generation is recommended for datasets used repeatedly, while lazy generation is simpler for initial experiments.

**Pre-Generate All Spectrograms:**
```bash
# Generate spectrograms for FSD50K dataset (default)
python -m audioloop.create_specs

# Generate spectrograms for UrbanSound8K dataset
python -m audioloop.create_specs --dataset urbansound8k

# Generate for AudioSet (if using entire dataset)
python -m audioloop.create_specs --dataset audioset

# With environment variable set (fallback when no explicit --dataset)
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.create_specs  # Uses environment variable

# Explicit parameter overrides environment variable
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.create_specs --dataset fsd50k  # Uses fsd50k despite env var
```

**Lazy Generation (Automatic):**
When training CSVs include an `audio_path` column (like those from `create_subset`), spectrograms are generated automatically during training if missing. No pre-processing needed.

```bash
# Create subset (includes audio_path column)
python -m audioloop.create_subset --dataset audioset --class-name "Dog" --max-samples 1000

# Train directly - specs generated as needed
python -m audioloop.train subsets/audioset_dog_1000.csv
```

**When to Use Each Approach:**
- **Pre-generation**: Large datasets used repeatedly, saves time across multiple experiments
- **Lazy generation**: Small subsets, quick experiments, changing spectrogram parameters

### Creating Initial Training Sets

```bash
# Create FSD50K training set (default)
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50

# Create UrbanSound8K training set
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --class-name siren --n 40

# Environment variable provides fallback when no explicit --dataset
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40  # Uses urbansound8k
python -m audioloop.utils.create_bootstrap_set --list-classes              # Lists urbansound8k classes

# Explicit parameter takes precedence over environment variable
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --class-name Drill  # Uses fsd50k
```

### Understanding Dataset Splits

Different datasets organize their data differently. AudioLoop now provides a unified interface to work with each dataset's natural split organization:

```bash
# Discover available splits for any dataset
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --list-splits
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --list-splits

# FSD50K has train/eval splits
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --list-splits
# Output:
# Available splits for fsd50k:
#   dev (default)
#   eval
# Usage: --split dev (or any split from the list above)

# UrbanSound8K treats all data as one split (designed for cross-validation)
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --list-splits
# Output:
# Available splits for urbansound8k:
#   all (default)
# Usage: --split all (or any split from the list above)

# Use specific splits when creating training sets
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --split eval --class-name Piano --n 100
```

**Key Points:**
- Each dataset defines its own valid split names
- Use `--list-splits` to discover what splits are available
- If no `--split` is provided, the dataset's default split is used
- Invalid split names will show clear error messages with valid options

```bash
# Create with custom parameters
python -m audioloop.utils.create_bootstrap_set --class-name Speech --n 60 --positive-pct 0.8 --output training_sets/training_set_v2.csv

# Create with custom seed for reproducible sampling
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50 --seed 123

# List available classes for FSD50K
python -m audioloop.utils.create_bootstrap_set --list-classes

# List available classes for UrbanSound8K  
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --list-classes

# Create with experiment organization
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40 --experiment myexp
```

**Bootstrap Set Parameters:**
```bash
--dataset {urbansound8k,fsd50k,audioset} # Dataset to use (overrides AUDIOLOOP_DATASET)
--class-name CLASS_NAME         # Class name for positive samples
--n N                          # Total number of samples (default: 40)
--positive-pct POSITIVE_PCT    # Percentage positive (default: 0.75)
--output OUTPUT                # Output CSV path
--run RUN                      # Run number for versioning
--seed SEED                    # Random seed for reproducibility
--experiment EXPERIMENT        # Experiment name for organization
--list-classes                 # List available classes for dataset

# Dataset-specific options
--metadata-csv METADATA_CSV    # Custom metadata CSV path
--audio-root AUDIO_ROOT        # Custom audio directory path
--output-dir OUTPUT_DIR        # Custom spectrogram output directory
--split {dev,eval}             # Dataset split (varies by dataset, use --list-splits to see options)
--list-splits                   # Show available splits for the selected dataset
```

## Training Models

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
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion plateau --patience 30
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion accuracy

# Plateau with accuracy floor (only count patience when accuracy >= threshold)
python -m audioloop.train training_sets/training_set_v1.csv --stopping-criterion plateau \
  --accuracy-floor 0.90 --patience 25

# Train with custom model
python -m audioloop.train training_sets/training_set_v1.csv --model-type simple_cnn

# Train with class weighting for imbalanced datasets
python -m audioloop.train training_sets/training_set_v1.csv --class-weighting adaptive
python -m audioloop.train training_sets/training_set_v1.csv --class-weighting 0.25

# List available models
python -m audioloop.train --list-models
```

**Training Parameters:**
```bash
--epochs 1000                  # Maximum training epochs
--batch-size 32               # Training batch size
--learning-rate 0.001         # Learning rate
--seed 42                     # Random seed for reproducibility
--stopping-criterion {accuracy,plateau}  # Stopping strategy (default: plateau)
--patience N                  # Epochs to wait for improvement (plateau)
--accuracy-floor 0.95         # Only count patience when accuracy >= threshold (plateau)
--model-type MODEL_TYPE       # Model architecture to use
--class-weighting {adaptive,0.0-1.0}  # Class weighting mode (see below)
--experiment EXPERIMENT       # Experiment name for organization
```

**Class Weighting Modes:**
- **Fixed 0.70** (default): Prevents model collapse with imbalanced training data
- **Adaptive** (`--class-weighting adaptive`): Calculates inverse frequency from current training set ratio each cycle
- **No weighting** (`--class-weighting null`): Treats all classes equally (can collapse to all-positive predictions)
- **Custom fixed** (`--class-weighting 0.25`): Set a specific target positive ratio (0.0-1.0)

## Active Learning

### Basic Active Learning Commands

```bash
# Run active learning cycle (auto-detects version from run number)
python -m audioloop.active_learning --class-name Drill --run-number 1

# Run with UrbanSound8K dataset
python -m audioloop.active_learning --dataset urbansound8k --class-name siren --run-number 1

# Run with custom seed for reproducible candidate selection
python -m audioloop.active_learning --class-name Drill --run-number 1 --seed 123

# Run with experiment name (uses outputs/myexp/ instead of outputs/)
python -m audioloop.active_learning --class-name Drill --run-number 1 --experiment myexp

# List available sound classes (respects AUDIOLOOP_DATASET environment variable)
python -m audioloop.active_learning --list-classes

# List UrbanSound8K classes
python -m audioloop.active_learning --dataset urbansound8k --list-classes
```

### Selection Strategies

```bash
# Entropy-based selection (default - focuses on high uncertainty examples)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode entropy
# Or simply omit --selection-mode to use default:
python -m audioloop.active_learning --class-name Drill --run-number 1

# High-confidence selection
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode confidence

# Mixed-entropy selection (experimental - samples 70% high / 20% medium / 10% low entropy)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode mixed_entropy

# Basic transition (starts with confidence, switches to entropy when criteria met)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition

# Basic transition with auto-calculated thresholds (recommended for imbalanced datasets)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition --auto-thresholds --estimated-positive-pct 0.10

# Basic transition with auto-thresholds using default estimate (5%)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition --auto-thresholds

# Custom basic transition thresholds (manual tuning)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition \
  --basic-transition-f1-threshold 0.25 --basic-transition-confidence-threshold 0.95 --basic-transition-variance-threshold 0.10
```

### Workflow Modes

```bash
# Production workflow (default - no ground truth columns, for real unlabeled datasets)
python -m audioloop.active_learning --class-name Drill --run-number 1

# Evaluation workflow (includes ground truth evaluation columns for research/evaluation)
python -m audioloop.active_learning --class-name Drill --run-number 1 --with-ground-truth
```

### Custom Parameters

```bash
# Run with explicit parameters
python -m audioloop.active_learning --class-name Speech --run-number 2 --total-candidates 20 --positive-pct 0.75 --min-confidence 0.85

# Custom candidate selection
python -m audioloop.active_learning --class-name dog_bark --run-number 1 \
  --total-candidates 30 --positive-pct 0.8 --min-confidence 0.9
```

**Active Learning Parameters:**
```bash
--total-candidates 20          # Number of samples to select (default: 20)
--positive-pct 0.75           # Percentage positive predictions (default: None, incompatible with mixed_entropy)
--min-confidence 0.8          # Minimum confidence threshold (default: 0.8)
--selection-mode {confidence,entropy,mixed_entropy,basic_transition}  # Selection strategy (default: entropy)
--with-ground-truth           # Include ground truth evaluation columns
--auto-thresholds             # Auto-calculate thresholds based on dataset characteristics
--estimated-positive-pct 0.05 # Estimated positive class percentage (default: 0.05)
--basic-transition-f1-threshold 0.25      # F1 threshold for transition (default: 0.2)
--basic-transition-confidence-threshold 0.95  # Confidence threshold for transition
--basic-transition-variance-threshold 0.10    # Variance threshold for transition
--experiment EXPERIMENT       # Experiment name for organization
--seed SEED                   # Random seed for reproducibility
```

## Human Labeling

### Web UI (Recommended)
```bash
# From your project directory, start the web labeling interface
cd ~/projects/my-classifier
python -m audioloop.webui

# Open browser to http://127.0.0.1:5000
# Select candidate file from dropdown (auto-populated from outputs/)
# Features: Visual interface, audio player, keyboard shortcuts, progress tracking
```

### Terminal Interface
```bash
# Interactive audio labeling tool (uses AUDIOLOOP_DATASET or default)
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# Explicit FSD50K dataset
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset fsd50k --audio-dir data/FSD50K/FSD50K.dev_audio

# UrbanSound8K with environment variable
AUDIOLOOP_DATASET=urbansound8k python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# Explicit UrbanSound8K dataset
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --dataset urbansound8k --audio-dir data/urbansound8k

# Resume labeling session (automatically finds unlabeled samples)
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv
```

**Labeling Tool Commands:**
- `1` or `y` - Label as positive
- `0` or `n` - Label as negative  
- `u` - Jump to next unlabeled
- `p` - Replay current audio
- `n` - Skip to next without labeling
- `b` - Go back to previous sample
- `j` - Jump to specific sample number
- `s` - Save progress
- `q` - Quit (prompts to save)
- `h` - Show help

## Label Management

### Production Workflow (Manual Labeling)
```bash
# Merge human labels back into training set (default experiment)
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Merge labels with explicit experiment name
python -m audioloop.merge_labels training_sets/myexp/training_set_v1.csv outputs/myexp/labeling_candidates_v1.csv --experiment myexp

# Merge with auto-generated output path
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv --experiment myexp
```

### Evaluation Workflow (Auto-Labeling for Research)
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

## Metrics and Analysis

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

## Automated Workflows

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

# Use fixed class weighting to stabilize performance
python -m audioloop.automated_workflow --class-name "Brass instrument" --cycles 10 --class-weighting 0.05 --evaluation-mode --auto-label

# Auto-stop when candidate F1 plateaus (label mode - good for auto-labeling)
python -m audioloop.automated_workflow --class-name Drill --cycle-stopping-strategy label --evaluation-mode --auto-label

# Auto-stop when candidate recall plateaus (search mode - good for finding rare positives)
python -m audioloop.automated_workflow --class-name Gunshot --cycle-stopping-strategy search --evaluation-mode --auto-label

# No auto-stopping (default - run fixed number of cycles)
python -m audioloop.automated_workflow --class-name Siren --cycles 20 --cycle-stopping-strategy none --evaluation-mode --auto-label
```

**Automated Workflow Parameters:**
```bash
--class-name CLASS_NAME       # Target class for binary classification (required)
--cycles N                    # Number of active learning cycles (default: 3)
--evaluation-mode             # Enable evaluation mode with ground truth access
--auto-label                  # Use ground truth for automatic labeling (requires --evaluation-mode)
--dataset {urbansound8k,fsd50k,audioset}  # Dataset choice
--epochs N                    # Max training epochs per cycle (default: 1000)
--candidates N                # Number of candidates per cycle (default: 50)
--positive-pct FLOAT          # Stratify candidates by prediction (0.0-1.0, default: None, incompatible with mixed_entropy)
--selection-mode {confidence,entropy,mixed_entropy,basic_transition}  # Selection strategy (default: entropy)
--cycle-stopping-strategy {none,label,search}  # Cycle stopping strategy (default: none)
--class-weighting {adaptive,null,0.0-1.0}  # Class weighting mode (default: 0.70)
--experiment EXPERIMENT       # Experiment name for organization
--seed SEED                   # Random seed for reproducibility
```

## Available Sound Classes

### UrbanSound8K Classes
| ID | Name | ID | Name |
|----|------|----|------|
| 0 | air_conditioner | 5 | engine_idling |
| 1 | car_horn | 6 | gun_shot |
| 2 | children_playing | 7 | jackhammer |
| 3 | dog_bark | 8 | siren |
| 4 | drilling | 9 | street_music |

### FSD50K Classes  
FSD50K has 200 classes. Common classes include:
- **Musical Instruments**: Piano, Guitar, Drum_kit, Violin, etc.
- **Human Sounds**: Speech, Laughter, Conversation, etc.
- **Vehicle Sounds**: Car, Aircraft, Train, etc.
- **Tools & Mechanical**: Drill, Hammer, etc.

```bash
# List all FSD50K classes
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --list-classes

# List UrbanSound8K classes
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --list-classes
```

### Class Selection
```bash
# By name (recommended)
python -m audioloop.active_learning --class-name siren
python -m audioloop.active_learning --class-name dog_bark
python -m audioloop.active_learning --class-name Drill  # FSD50K

# By ID (UrbanSound8K only)
python -m audioloop.active_learning --class-id 8  # siren
python -m audioloop.active_learning --class-id 3  # dog_bark
```

## File Outputs

### Generated Files
Each active learning cycle generates:
- `outputs/predictions_v{N}.csv` - Model predictions on full dataset
- `outputs/labeling_candidates_v{N}.csv` - Samples for human review
- `outputs/model_v{N}.pt` - Trained model

After human labeling and merging:
- `training_sets/training_set_v{N+1}.csv` - Expanded training set

### File Formats

**Training Set CSV:**
```csv
filepath,label,run
data/all_specs/100032-3-0-0.pt,1,1
data/all_specs/100263-2-0-117.pt,0,1
```

**Predictions CSV (Production Mode):**
```csv
filename,prediction,predicted_class,confidence,entropy,prob_negative,prob_positive,original_class,fold,filepath
```

**Predictions CSV (Evaluation Mode with --with-ground-truth):**
```csv
filename,ground_truth,prediction,predicted_class,confidence,entropy,prob_negative,prob_positive,correct,original_class,fold,filepath
```

**Candidates CSV (For Human Labeling):**
```csv
filename,prediction,predicted_class,confidence,needs_human_label,entropy,prob_negative,prob_positive,original_class,fold,filepath
```

## Code Quality and Development

```bash
# Format and lint code
ruff check src/audioloop/
ruff format src/audioloop/

# List available models
python -m audioloop.train --list-models

# List available datasets
python -c "from audioloop.datasets.registry import list_available_datasets; print(list_available_datasets())"
```

## Python API Examples

### Configuration System
```python
from audioloop.config import AudioLoopConfig

# Create configuration
config = AudioLoopConfig(experiment_name="test", dataset="urbansound8k")

# Access organized paths
print(config.output_dir)         # outputs/test/
print(config.training_sets_dir)  # training_sets/test/
print(config.specs_dir)         # data/all_specs/

# Generate versioned paths
model_path = config.get_model_path(1)        # outputs/test/model_v1.pt
pred_path = config.get_predictions_path(1)   # outputs/test/predictions_v1.csv
```

### Core Functions
```python
from audioloop.training_core import run_training
from audioloop.active_learning_core import run_active_learning_cycle
from audioloop.merge_labels import merge_training_sets

# Train model
accuracy = run_training(
    config=config,
    labels_file='training_sets/training_set_v1.csv',
    version=1,
    max_epochs=1000,
    seed=42
)

# Run active learning
predictions_file, candidates_file = run_active_learning_cycle(
    config=config,
    positive_class_name="siren",
    run_number=1
)

# Merge labels
new_training_set = merge_training_sets(
    "training_sets/training_set_v1.csv",
    "outputs/labeling_candidates_v1.csv",
    config=config
)
print(f"Created: {new_training_set}")
```

## Best Practices and Tips

### Labeling Quality
1. **Listen completely** - Let audio play fully before deciding
2. **Be consistent** - Apply same criteria throughout session
3. **Skip if unclear** - Better to skip than guess incorrectly
4. **Take breaks** - Avoid ear fatigue every 50-100 samples
5. **Focus on quality** - Confident labels are more valuable than quantity

### Training Set Creation
1. **Start balanced** - Use 70-80% positive samples for initial training
2. **Sufficient samples** - Aim for 40-60 samples total to start
3. **Verify classes** - Use `--list-classes` to see available options
4. **Use seeds** - Add `--seed 42` for reproducible training sets
5. **Check output** - Verify CSV format matches expected structure

### Model Training
1. **Start small** - Begin with 20-50 samples per class
2. **Monitor accuracy** - Should reach 95%+ on training set
3. **Check convergence** - Training should complete in <500 epochs
4. **Use experiments** - Organize with `--experiment` for clean separation

### Active Learning Strategy
1. **Start with confidence** - Use confidence-based selection initially
2. **Switch to entropy** - When model becomes overconfident
3. **Use basic transition** - Let system automatically switch strategies
4. **Balance classes** - Maintain ~75% positive, 25% negative
5. **Iterate quickly** - Short cycles (20-50 samples) work better than long ones

## Troubleshooting

### Common Issues and Solutions

**Spectrograms not found:**
```bash
python -m audioloop.create_specs  # Regenerate spectrograms
```

**No training set exists:**
```bash
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40  # Create initial set
```

**Invalid class name:**
```bash
python -m audioloop.utils.create_bootstrap_set --list-classes  # See available classes
```

**Not enough samples for class:**
```bash
python -m audioloop.utils.create_bootstrap_set --class-name rare_class --n 10  # Reduce sample count
```

**Model not found:**
```bash
python -m audioloop.train training_sets/training_set_v1.csv  # Train first
```

**Audio won't play:**
```bash
python -m audioloop.label_audio file.csv --audio-dir /full/path/to/audio
```

**Version mismatch:**
```bash
python -m audioloop.active_learning --run-number 2 --model outputs/model_v2.pt
```

**Environment variable issues:**
```bash
export AUDIOLOOP_DATASET=fsd50k        # Set valid dataset
unset AUDIOLOOP_DATASET                # Remove invalid setting
echo $AUDIOLOOP_DATASET                # Check current setting
```

### Performance Issues
- **Slow training**: Reduce batch size or use smaller model
- **Out of memory**: Lower batch size (`--batch-size 16`) or use CPU
- **Slow audio loading**: Ensure audio files are local, not networked
- **Training not converging**: Check training set balance and size

### Dataset Issues
- **Wrong dataset format**: Use correct `--dataset` parameter
- **Missing audio files**: Check audio directory paths
- **Inconsistent file paths**: Use absolute paths or correct relative paths

## Quick Reference

### Essential Commands
```bash
# Setup
python -m audioloop.create_specs
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40

# Manual workflow
python -m audioloop.train training_sets/training_set_v1.csv
python -m audioloop.active_learning --class-name siren --run-number 1
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Automated workflow
python -m audioloop.automated_workflow --class-name siren --cycles 3 --evaluation-mode --auto-label

# Analysis
python -m audioloop.track_metrics --plot
```

### Next Steps
See [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for complete workflow patterns and [DEV_GUIDE.md](DEV_GUIDE.md) for development information.