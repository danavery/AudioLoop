# AudioLoop Usage Guide

Quick reference for AudioLoop commands and practical usage patterns.

## Automated Workflow (Recommended)

The `automated_workflow.py` script provides a simple way to run complete active learning cycles with minimal manual intervention.

### Quick Start
```bash
# Prerequisites (one-time setup)
python -m audioloop.create_all_specs
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40

# Production workflow (pause for human labeling)
python -m audioloop.automated_workflow --class-name dog_bark --cycles 2

# Evaluation workflow (auto-labeling for testing)
python -m audioloop.automated_workflow --class-name siren --cycles 3 --evaluation-mode --auto-label
```

### Common Usage Patterns
```bash
# Production workflow with human labeling
python -m audioloop.automated_workflow --class-name dog_bark --cycles 2

# Evaluation workflow for testing
python -m audioloop.automated_workflow --class-name siren --cycles 3 --evaluation-mode --auto-label

# Custom training parameters
python -m audioloop.automated_workflow --class-name gun_shot --cycles 3 --epochs 500 --batch-size 64

# Custom active learning parameters
python -m audioloop.automated_workflow --class-name jackhammer --cycles 2 --candidates 100 --positive-pct 0.8

# FSD50K evaluation workflow
python -m audioloop.automated_workflow --class-name Drill --cycles 2 --dataset fsd50k --evaluation-mode --auto-label
```

### Parameters
- `--class-name`: Target class for binary classification (required)
- `--cycles`: Number of active learning cycles (default: 3)
- `--evaluation-mode`: Enable evaluation mode with ground truth access
- `--auto-label`: Use ground truth for automatic labeling (requires --evaluation-mode)
- `--dataset`: Dataset choice (`urbansound8k` or `fsd50k`)
- `--epochs`: Max training epochs per cycle (default: 1000)
- `--candidates`: Number of candidates per cycle (default: 50)
- `--positive-pct`: Target percentage of positive samples (default: 0.75)

## Manual Command Reference

### Configuration System

AudioLoop uses a unified configuration system with proper precedence:

```bash
# Environment variables (used as fallbacks when no explicit parameters)
export AUDIOLOOP_DATASET=urbansound8k      # Default dataset
export AUDIOLOOP_DATA_ROOT=/custom/data    # Custom data directory
export AUDIOLOOP_OUTPUT_ROOT=/custom/out   # Custom output directory

# Configuration precedence:
# 1. Explicit CLI parameters (highest priority)
# 2. Environment variables (fallback)
# 3. Default values (lowest priority)
```

### Data Preparation
```bash
# Generate spectrograms for entire FSD50K dataset (one-time setup)
python -m audioloop.create_all_specs

# Generate spectrograms for UrbanSound8K dataset
python -m audioloop.create_all_specs --dataset urbansound8k

# With environment variable set (fallback when no explicit --dataset)
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.create_all_specs  # Uses environment variable

# Explicit parameter overrides environment variable
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.create_all_specs --dataset fsd50k  # Uses fsd50k despite env var
```

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

# Create with custom parameters
python -m audioloop.utils.create_bootstrap_set --class-name Speech --n 60 --positive-pct 0.8 --output training_sets/training_set_v2.csv

# List available classes for FSD50K
python -m audioloop.utils.create_bootstrap_set --list-classes

# List available classes for UrbanSound8K  
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --list-classes

### Experiment Organization

AudioLoop automatically organizes files by experiment:

```bash
# Default behavior (no experiment name)
python -m audioloop.train training_sets/training_set_v1.csv
# Output: outputs/model_v1.pt

# With experiment name
python -m audioloop.train training_sets_myexp/training_set_v1.csv --experiment myexp
# Output: outputs_myexp/model_v1.pt

# All commands support --experiment parameter
python -m audioloop.active_learning --class-name siren --run-number 1 --experiment myexp
# Uses: training_sets_myexp/ and outputs_myexp/
```

### Configuration in Python Code

For programmatic usage, use the unified configuration system:

```python
from audioloop.config import AudioLoopConfig

# Create configuration
config = AudioLoopConfig(experiment_name="test", dataset="urbansound8k")

# Access organized paths
print(config.output_dir)         # outputs_test/
print(config.training_sets_dir)  # training_sets_test/
print(config.specs_dir)         # data/all_specs/

# Generate versioned paths
model_path = config.get_model_path(1)        # outputs_test/model_v1.pt
pred_path = config.get_predictions_path(1)   # outputs_test/predictions_v1.csv
```
python -m audioloop.utils.create_bootstrap_set --class-name siren --metadata-csv /path/to/UrbanSound8K.csv --audio-root /path/to/audio
```

### Training Models
```bash
# Train model (version auto-detected from filename)
python -m audioloop.train training_sets/training_set_v1.csv

# Train with explicit version and parameters
python -m audioloop.train training_sets/training_set_v1.csv -v 1 --epochs 500 --batch-size 64
```

### Active Learning Workflow
```bash
# Production workflow (default - no ground truth access)
python -m audioloop.active_learning --class-name siren --run-number 1

# Evaluation workflow (includes ground truth for comprehensive metrics)
python -m audioloop.active_learning --class-name siren --run-number 1 --with-ground-truth

# Run with basic transition strategy and auto-calculated thresholds (recommended for imbalanced datasets)
python -m audioloop.active_learning --class-name siren --run-number 1 --selection-mode basic_transition --auto-thresholds --estimated-positive-pct 0.10

# Run with basic transition strategy using default thresholds
python -m audioloop.active_learning --class-name siren --run-number 1 --selection-mode basic_transition

# Run with auto-thresholds using default estimate (5%)
python -m audioloop.active_learning --class-name siren --run-number 1 --selection-mode basic_transition --auto-thresholds

# Run with custom basic transition thresholds
python -m audioloop.active_learning --class-name siren --run-number 1 --selection-mode basic_transition \
  --basic-transition-f1-threshold 0.25 --basic-transition-confidence-threshold 0.95 --basic-transition-variance-threshold 0.10

# Run with explicit parameters
python -m audioloop.active_learning --class-name dog_bark --run-number 2 --total-candidates 20 --positive-pct 0.75 --min-confidence 0.85

# List available sound classes
python -m audioloop.active_learning --list-classes

# FSD50K dataset
python -m audioloop.active_learning --dataset fsd50k --class-name Drill --run-number 1
```

### Workflow Modes

AudioLoop supports two distinct workflow modes:

#### Production Mode (Default)
**Use case**: Real-world deployment with truly unlabeled datasets
```bash
# Production active learning (no ground truth columns)
python -m audioloop.active_learning --class-name siren --run-number 1

# Manual labeling using web UI (recommended)
cd webui && python app.py  # Load: outputs/labeling_candidates_v1.csv

# Or terminal interface
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k

# Track prediction and confidence metrics only
python -m audioloop.track_metrics --plot
```

#### Evaluation Mode (Research/Testing)
**Use case**: Research and testing with known datasets
```bash
# Evaluation active learning (includes ground truth columns)
python -m audioloop.active_learning --class-name siren --run-number 1 --with-ground-truth

# Auto-label using ground truth (for rapid testing)
python -m audioloop.auto_label_candidates outputs/labeling_candidates_v1.csv

# Track comprehensive evaluation metrics (F1, precision, recall, accuracy)
python -m audioloop.track_metrics --plot
```

### Human Labeling
```bash
# Interactive audio labeling tool
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k

# Resume labeling session
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv  # Automatically finds unlabeled samples
```

### Label Management
```bash
# Merge human labels back into training set
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Creates training_sets/training_set_v2.csv automatically
```

## Class Selection

### By Name (Recommended)
```bash
python -m audioloop.active_learning --class-name siren
python -m audioloop.active_learning --class-name dog_bark
python -m audioloop.active_learning --class-name gun_shot
python -m audioloop.active_learning --class-name car_horn
```

### By ID
```bash
python -m audioloop.active_learning --class-id 8  # siren
python -m audioloop.active_learning --class-id 3  # dog_bark
python -m audioloop.active_learning --class-id 6  # gun_shot
```

### Available Classes

#### UrbanSound8K Classes
| ID | Name | ID | Name |
|----|------|----|------|
| 0 | air_conditioner | 5 | engine_idling |
| 1 | car_horn | 6 | gun_shot |
| 2 | children_playing | 7 | jackhammer |
| 3 | dog_bark | 8 | siren |
| 4 | drilling | 9 | street_music |

#### FSD50K Classes
FSD50K has 200 classes. Use `--list-classes` to see all available classes:
```bash
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --list-classes
```

Common FSD50K classes include: Drill, Gunshot_and_gunfire, Siren, Dog, Cat, Music, Speech, etc.

## Common Workflows

### Complete 3-Cycle Example
```bash
# Optional: Set dataset once for entire workflow
export AUDIOLOOP_DATASET=urbansound8k  # or fsd50k

# Step 0: Create initial training set
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40 --output training_sets/training_set_v1.csv

# Cycle 1: Initial training
python -m audioloop.train training_sets/training_set_v1.csv
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Cycle 2: Improved model
python -m audioloop.train training_sets/training_set_v2.csv
python -m audioloop.active_learning --class-name siren --model outputs/model_v2.pt
python -m audioloop.label_audio outputs/labeling_candidates_v2.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v2.csv outputs/labeling_candidates_v2.csv

# Cycle 3: Final iteration
python -m audioloop.train training_sets/training_set_v3.csv
python -m audioloop.active_learning --class-name siren --model outputs/model_v3.pt
```

### Starting from Scratch (New Class)
```bash
# Create initial training set with balanced samples
python -m audioloop.utils.create_bootstrap_set --class-name gun_shot --n 50 --positive-pct 0.7

# Train initial model
python -m audioloop.train training_sets/training_set_v1.csv

# Begin active learning
python -m audioloop.active_learning --class-name gun_shot --model outputs/model_v1.pt
```

### Multi-Dataset Workflow
```bash
# Create training sets for different datasets (explicit approach)
python -m audioloop.utils.create_bootstrap_set --dataset urbansound8k --class-name siren --n 40
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --class-name Siren --n 60

# Simplified with environment variable
export AUDIOLOOP_DATASET=urbansound8k
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40
python -m audioloop.create_all_specs

export AUDIOLOOP_DATASET=fsd50k  
python -m audioloop.utils.create_bootstrap_set --class-name Siren --n 60
python -m audioloop.create_all_specs

# Compare performance across datasets
python -m audioloop.train training_sets/training_set_v1.csv  # UrbanSound8K
python -m audioloop.train training_sets/training_set_v2.csv  # FSD50K
```

### Automated Workflow
```bash
# Run complete workflow with simulated labeling
python example_workflow.py --class-name siren --cycles 2

# Run with manual human labeling
python example_workflow.py --class-name dog_bark --cycles 3 --no-simulate
```

### Batch Processing Multiple Classes
```bash
# Process multiple classes
for class_name in siren dog_bark gun_shot; do
    python -m audioloop.active_learning --class-name $class_name --model outputs/model_v1.pt
done

# Process all classes by ID
for class_id in {0..9}; do
    python -m audioloop.active_learning --class-id $class_id --model outputs/model.pt
done
```

## Python API Examples

### Basic Active Learning
```python
from audioloop.active_learning import run_active_learning_for_class

# Simple approach
predictions_file, candidates_file = run_active_learning_for_class(
    positive_class_name="siren",
    model_path="outputs/model_v1.pt",
    run_number=1
)
```

### Training Models
```python
from audioloop.training_core import run_training

accuracy = run_training(
    labels_file='training_sets/training_set_v1.csv',
    max_epochs=1000,
    seed=42
)
```

### Label Management
```python
from audioloop.merge_labels import merge_training_sets

new_training_set = merge_training_sets(
    "training_sets/training_set_v1.csv",
    "outputs/labeling_candidates_v1.csv"  # With human labels filled in
)
print(f"Created: {new_training_set}")
```

### Creating Initial Training Sets
```python
from audioloop.utils.start_labeling import create_training_set

# Create for FSD50K (default)
create_training_set(class_name="Drill", n=50)
create_training_set(class_name="Speech", output_path="training_sets/speech_v1.csv")

# Create for UrbanSound8K
create_training_set(
    class_name="dog_bark", 
    dataset_name="urbansound8k", 
    n=15,
    positive_percentage=0.8
)

# Create with custom parameters
create_training_set(
    class_name="siren",
    dataset_name="urbansound8k", 
    n=60,
    positive_percentage=0.75,
    output_path="training_sets/siren_training.csv",
    run=1
)
```

## Advanced Parameters

### Environment Variable
```bash
# Set default dataset for all commands
export AUDIOLOOP_DATASET=fsd50k        # Default dataset (optional)
export AUDIOLOOP_DATASET=urbansound8k  # Use UrbanSound8K as default
unset AUDIOLOOP_DATASET                # Return to system default (fsd50k)

# Invalid values show helpful error messages
export AUDIOLOOP_DATASET=invalid_name  # Will show error with supported options
```

### Initial Training Set Options
```bash
--dataset {urbansound8k,fsd50k} # Dataset to use (overrides AUDIOLOOP_DATASET)
--class-name CLASS_NAME         # Class name for positive samples
--n N                          # Total number of samples (default: 40)
--positive-pct POSITIVE_PCT    # Percentage positive (default: 0.75)
--output OUTPUT                # Output CSV path
--run RUN                      # Run number for versioning
--seed SEED                    # Random seed for reproducibility
--list-classes                 # List available classes for dataset

# Dataset-specific options
--metadata-csv METADATA_CSV    # Custom metadata CSV path
--audio-root AUDIO_ROOT        # Custom audio directory path
--output-dir OUTPUT_DIR        # Custom spectrogram output directory
--split {dev,eval}             # FSD50K dataset split (default: dev)
```

### Active Learning Options
```bash
--total-candidates 30           # Number of samples to select (default: 20)
--positive-pct 0.8             # Percentage positive predictions (default: 0.75)
--min-confidence 0.9           # Minimum confidence threshold (default: 0.8)
--negative-name "background"   # Custom negative class name
--selection-mode basic_transition  # Use basic transition strategy
--auto-thresholds              # Auto-calculate thresholds based on dataset characteristics
--estimated-positive-pct 0.05  # Estimated positive class percentage (default: 0.05)
--basic-transition-f1-threshold 0.25  # F1 threshold for transition (default: 0.2)
```

### Training Options
```bash
--epochs 1000                  # Maximum training epochs
--batch-size 32               # Training batch size
--learning-rate 0.001         # Learning rate
--seed 42                     # Random seed for reproducibility
```

### Labeling Tool Commands
- `1` or `y` - Label as positive
- `0` or `n` - Label as negative
- `u` - Jump to next unlabeled
- `p` - Replay current audio
- `s` - Save progress
- `q` - Quit (prompts to save)
- `h` - Show help

## File Outputs

Each active learning cycle generates:
- `outputs/binary_labels_v{N}.csv` - Binary labels for target class
- `outputs/predictions_v{N}.csv` - Model predictions on full dataset
- `outputs/labeling_candidates_v{N}.csv` - Samples for human review

After human labeling and merging:
- `training_sets/training_set_v{N+1}.csv` - Expanded training set

## Quality Control Tips

### Labeling Best Practices
1. **Listen completely** - Let audio play fully before deciding
2. **Be consistent** - Apply same criteria throughout session
3. **Skip if unclear** - Better to skip than guess incorrectly
4. **Take breaks** - Avoid ear fatigue every 50-100 samples
5. **Focus on quality** - Confident labels are more valuable than quantity

### Training Set Creation Tips
1. **Start balanced** - Use 70-80% positive samples for initial training
2. **Sufficient samples** - Aim for 40-60 samples total to start
3. **Verify classes** - Use `--list-classes` to see available options
4. **Use seeds** - Add `--seed 42` for reproducible training sets
5. **Check output** - Verify CSV format matches expected structure

### Model Training Tips
1. **Start small** - Begin with 10-20 samples per class
2. **Monitor accuracy** - Should reach 95%+ on training set
3. **Check convergence** - Training should complete in <500 epochs
4. **Validate results** - Use `python -m audioloop.test_model` to verify

### Active Learning Tips
1. **High confidence first** - Start with threshold 0.8-0.9
2. **Balance classes** - Maintain ~75% positive, 25% negative
3. **Review mistakes** - High-confidence errors are most valuable
4. **Iterate quickly** - Short cycles (20 samples) work better than long ones

## Troubleshooting

### Common Issues
```bash
# Spectrograms not found
python -m audioloop.create_all_specs  # Regenerate spectrograms

# No training set exists
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 40  # Create initial set

# Invalid class name
python -m audioloop.utils.create_bootstrap_set --list-classes  # See available classes

# Not enough samples for class
python -m audioloop.utils.create_bootstrap_set --class-name rare_class --n 10  # Reduce sample count

# Model not found
python -m audioloop.train training_sets/training_set_v1.csv  # Train first

# Audio won't play
python -m audioloop.label_audio file.csv --audio-dir /full/path/to/audio

# Version mismatch
python -m audioloop.active_learning --run-number 2 --model outputs/model_v2.pt

# Wrong dataset format
python -m audioloop.utils.create_bootstrap_set --dataset fsd50k --class-name Drill  # Use correct dataset

# Environment variable issues
export AUDIOLOOP_DATASET=fsd50k        # Set valid dataset
unset AUDIOLOOP_DATASET                # Remove invalid setting
echo $AUDIOLOOP_DATASET                # Check current setting
```

### Performance Issues
- **Slow training**: Reduce batch size or use CPU with `--device cpu`
- **Out of memory**: Lower batch size or use smaller model
- **Slow audio loading**: Ensure audio files are local, not networked

## Integration Examples

### Custom Workflows
```python
from audioloop.utils.start_labeling import create_training_set
from audioloop.active_learning import run_active_learning_for_class

# Multi-class experiment with initial training sets
classes = ["siren", "dog_bark", "gun_shot"]
results = {}

for i, class_name in enumerate(classes, 1):
    # Create initial training set
    create_training_set(
        class_name=class_name,
        n=50,
        positive_percentage=0.75,
        output_path=f"training_sets/training_set_{class_name}_v1.csv"
    )
    
    # Run active learning
    predictions, candidates = run_active_learning_for_class(
        positive_class_name=class_name,
        model_path="outputs/model.pt",
        run_number=1
    )
    results[class_name] = candidates

print("Generated candidates for:", list(results.keys()))
```

### Custom Selection Strategies
```python
from audioloop.active_learning_core import select_candidates_for_labeling

# High-confidence selection
candidates = select_candidates_for_labeling(
    predictions_file="outputs/predictions.csv",
    total_candidates=50,
    positive_percentage=0.6,
    min_confidence=0.95,
    candidates_csv="outputs/high_conf_candidates.csv"
)
```

See [README.md](README.md) for project overview and [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) for versioned workflow patterns.