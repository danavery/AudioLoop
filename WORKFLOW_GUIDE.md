# AudioLoop Versioned Workflow Guide

Complete reference for AudioLoop's versioned active learning workflow patterns.

## Automated Workflow (Recommended)

The `automated_workflow.py` script automates the complete active learning cycle, eliminating manual steps and reducing complexity.

### Quick Start
```bash
# Prerequisites (one-time setup)
python -m audioloop.create_all_specs
python -m audioloop.utils.start_labeling --class-name siren --n 40

# Production workflow (pause for human labeling)
python -m audioloop.automated_workflow --class-name dog_bark --cycles 2

# Evaluation workflow (auto-labeling for testing)
python -m audioloop.automated_workflow --class-name siren --cycles 3 --evaluation-mode --auto-label
```

### What the Automated Workflow Does
1. **Train Model**: Uses existing training set to create `model_v{N}.pt`
2. **Active Learning**: Runs inference to generate `labeling_candidates_v{N}.csv`
3. **Label Candidates**: Either auto-labels (evaluation mode) or pauses for human labeling (production mode)
4. **Merge Labels**: Creates expanded `training_set_v{N+1}.csv`
5. **Repeat**: Continues for specified number of cycles

### Workflow Modes
- **Production Mode** (default): Pauses for human labeling, no ground truth access
- **Evaluation Mode**: Uses auto-labeling with ground truth for testing and research

### Advanced Usage
```bash
# Custom training parameters
python -m audioloop.automated_workflow --class-name gun_shot --cycles 3 --epochs 500 --batch-size 64

# Custom active learning parameters
python -m audioloop.automated_workflow --class-name jackhammer --cycles 2 --candidates 100 --positive-pct 0.8

# FSD50K evaluation workflow
python -m audioloop.automated_workflow --class-name Drill --cycles 2 --dataset fsd50k --evaluation-mode --auto-label
```

### Parameters
- `--class-name`: Target class for binary classification
- `--cycles`: Number of active learning cycles to run
- `--evaluation-mode`: Enable evaluation mode with ground truth access
- `--auto-label`: Use ground truth for automatic labeling (requires --evaluation-mode)
- `--dataset`: Dataset choice (`urbansound8k` or `fsd50k`)
- `--epochs`: Max training epochs per cycle
- `--candidates`: Number of candidates to select per cycle
- `--positive-pct`: Target percentage of positive samples

## Manual Workflow (Step-by-step)

For fine-grained control or educational purposes, you can run each step manually:

## Configuration and Experiment Organization

AudioLoop uses a unified configuration system that automatically organizes files by experiment:

### Default Organization (No Experiment Name)
- **Models**: `outputs/model_v{N}.pt`
- **Training sets**: `training_sets/training_set_v{N}.csv`
- **Predictions**: `outputs/predictions_v{N}.csv`
- **Candidates**: `outputs/labeling_candidates_v{N}.csv`

### Experiment Organization (With --experiment parameter)
- **Models**: `outputs_{experiment}/model_v{N}.pt`
- **Training sets**: `training_sets_{experiment}/training_set_v{N}.csv`
- **Predictions**: `outputs_{experiment}/predictions_v{N}.csv`
- **Candidates**: `outputs_{experiment}/labeling_candidates_v{N}.csv`

### Configuration Precedence
AudioLoop follows standard configuration patterns:
1. **Explicit CLI parameters** (highest priority)
2. **Environment variables** (fallback when no explicit value)
3. **Default values** (lowest priority)

```bash
# Environment variables provide defaults
export AUDIOLOOP_DATASET=urbansound8k
export AUDIOLOOP_OUTPUT_ROOT=/custom/outputs

# Explicit parameters override environment variables
python -m audioloop.train training_sets/training_set_v1.csv --dataset fsd50k
# Uses fsd50k despite environment variable
```

## Standard Workflow Pattern

### Complete 3-Cycle Example (Dog Bark Detection)

#### Default Organization (No Experiment)
```bash
# === CYCLE 1: Initial Training ===
# 1. Train initial model (auto-detects v1 from filename)
python -m audioloop.train training_sets/training_set_v1.csv
# → Creates: outputs/model_v1.pt

# 2. Run active learning (creates complete inference record and candidates)
python -m audioloop.active_learning --class-name dog_bark --run-number 1
# → Creates: outputs/predictions_v1.csv, outputs/labeling_candidates_v1.csv

# 3. Human labeling
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k

# 4. Merge labels (auto-creates v2)
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv
# → Creates: training_sets/training_set_v2.csv
```

#### Organized Experiment (Recommended)
```bash
# === ORGANIZED EXPERIMENT: dog_bark_exp ===
# All files automatically organized under experiment directories

# 1. Train initial model with experiment organization
python -m audioloop.train training_sets_dog_bark_exp/training_set_v1.csv --experiment dog_bark_exp
# → Creates: outputs_dog_bark_exp/model_v1.pt

# 2. Run active learning with experiment organization
python -m audioloop.active_learning --class-name dog_bark --run-number 1 --experiment dog_bark_exp
# → Creates: outputs_dog_bark_exp/predictions_v1.csv, outputs_dog_bark_exp/labeling_candidates_v1.csv

# 3. Human labeling (automatically uses correct experiment paths)
python -m audioloop.label_audio outputs_dog_bark_exp/labeling_candidates_v1.csv --audio-dir data/urbansound8k

# 4. Merge labels (creates organized training set)
python -m audioloop.merge_labels training_sets_dog_bark_exp/training_set_v1.csv outputs_dog_bark_exp/labeling_candidates_v1.csv
# → Creates: training_sets_dog_bark_exp/training_set_v2.csv

# === CYCLE 2: Improved Model ===
python -m audioloop.train training_sets_dog_bark_exp/training_set_v2.csv --experiment dog_bark_exp
# → Creates: outputs_dog_bark_exp/model_v2.pt

python -m audioloop.active_learning --class-name dog_bark --run-number 2 --experiment dog_bark_exp
# → Creates: outputs_dog_bark_exp/predictions_v2.csv, outputs_dog_bark_exp/labeling_candidates_v2.csv
```

## Version Auto-Detection

AudioLoop automatically detects versions from filenames:

### Training Models
```bash
# Version auto-detected from training_set_v{N}.csv
python -m audioloop.train training_sets/training_set_v1.csv
# → Creates: outputs/model_v1.pt

python -m audioloop.train training_sets/training_set_v2.csv  
# → Creates: outputs/model_v2.pt

# Override auto-detection
python -m audioloop.train training_sets/training_set_v1.csv -v 5
# → Creates: outputs/model_v5.pt
```

### Active Learning Cycles
```bash
# Version auto-detected from run number (finds corresponding model)
python -m audioloop.active_learning --class-name siren --run-number 1
# → Uses: outputs/model_v1.pt
# → Creates: outputs/predictions_v1.csv, outputs/labeling_candidates_v1.csv

python -m audioloop.active_learning --class-name siren --run-number 2
# → Uses: outputs/model_v2.pt
# → Creates: outputs/predictions_v2.csv, outputs/labeling_candidates_v2.csv

# Explicit model specification (overrides auto-detection)
python -m audioloop.active_learning --class-name siren --run-number 1 \
    --model outputs/custom_model.pt
```

## Clean Architecture

**DESIGN**: AudioLoop maintains clean separation between inference and analysis:

### Active Learning Pipeline (Focused Inference)
- **Input**: Trained model + dataset metadata
- **Process**: Runs inference on ALL available files, creates complete prediction record
- **Output**: Predictions with confidence + selected candidates for human labeling
- **Focus**: Lean, fast inference engine without statistical analysis

### Metrics Pipeline (Comprehensive Analysis)
- **Input**: Prediction files from multiple active learning iterations
- **Process**: Calculate trends, accuracy, F1, precision, recall across versions
- **Output**: Performance analysis, plots, and insights
- **Focus**: Rich evaluation and progress tracking

### Example: Clean Data Flow
```bash
# Version auto-detected from model filename
python -m audioloop.active_learning --class-name siren --model outputs/model_v2.pt
# → Creates: outputs/predictions_v2.csv, outputs/labeling_candidates_v2.csv

# Or specify run number explicitly
python -m audioloop.active_learning --class-name siren --run-number 3
# → Uses: outputs/model_v3.pt (must exist)
# → Creates: outputs/predictions_v3.csv, outputs/labeling_candidates_v3.csv
```

### Label Merging
```bash
# Next version auto-created
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv
# → Creates: training_sets/training_set_v2.csv

python -m audioloop.merge_labels training_sets/training_set_v2.csv outputs/labeling_candidates_v2.csv
# → Creates: training_sets/training_set_v3.csv
```

## Workflow Patterns

### Pattern 1: Sequential Versioning (Recommended)
```bash
# Always increment versions sequentially
v1 → v2 → v3 → v4...

# Training progression:
training_set_v1.csv → model_v1.pt → predictions_v1.csv → training_set_v2.csv
training_set_v2.csv → model_v2.pt → predictions_v2.csv → training_set_v3.csv
training_set_v3.csv → model_v3.pt → predictions_v3.csv → training_set_v4.csv
```

### Pattern 2: Branching for Experiments
```bash
# Create experimental branches
python -m audioloop.train training_sets/training_set_v2.csv -v 2a
python -m audioloop.train training_sets/training_set_v2.csv -v 2b --epochs 1000

# Compare results
python -m audioloop.active_learning --model outputs/model_v2a.pt --run-number 2a
python -m audioloop.active_learning --model outputs/model_v2b.pt --run-number 2b
```

### Pattern 3: Class-Specific Organization (Recommended)
```bash
# Use experiment names to organize by class - no file conflicts!
python -m audioloop.active_learning --class-name siren --run-number 1 --experiment siren_exp
# → outputs_siren_exp/labeling_candidates_v1.csv

python -m audioloop.active_learning --class-name dog_bark --run-number 1 --experiment dog_bark_exp
# → outputs_dog_bark_exp/labeling_candidates_v1.csv

# Clean separation - no overwrites or manual renaming needed!
```

## File Naming Reference

| File Type | Default Pattern | Experiment Pattern | Auto-Generated | Example |
|-----------|-----------------|-------------------|----------------|---------|
| Training Set | `training_sets/training_set_v{N}.csv` | `training_sets_{exp}/training_set_v{N}.csv` | By merge_labels | `training_sets_myexp/training_set_v1.csv` |
| Model | `outputs/model_v{N}.pt` | `outputs_{exp}/model_v{N}.pt` | By train | `outputs_myexp/model_v1.pt` |
| Predictions | `outputs/predictions_v{N}.csv` | `outputs_{exp}/predictions_v{N}.csv` | By active_learning | `outputs_myexp/predictions_v1.csv` |
| Candidates | `outputs/labeling_candidates_v{N}.csv` | `outputs_{exp}/labeling_candidates_v{N}.csv` | By active_learning | `outputs_myexp/labeling_candidates_v1.csv` |

## Version Tracking Best Practices

### 1. Keep All Versions
```bash
# Don't delete intermediate versions - experiment organization makes this clean
ls outputs_siren_exp/
model_v1.pt  model_v2.pt  model_v3.pt  # Keep all for comparison

ls training_sets_siren_exp/
training_set_v1.csv  training_set_v2.csv  training_set_v3.csv  # Track progression

# Multiple experiments stay organized
ls -d outputs_*/
outputs_siren_exp/  outputs_dog_bark_exp/  outputs_gun_shot_exp/
```

### 2. Document Changes
```bash
# Add notes about what changed between versions
# training_set_v1.csv: Initial 20 samples
# training_set_v2.csv: Added 15 high-confidence corrections
# training_set_v3.csv: Added 20 boundary cases
```

### 3. Performance Tracking
```bash
# Compare model performance across versions
python -c "
import pandas as pd
v1 = pd.read_csv('outputs/predictions_v1.csv')
v2 = pd.read_csv('outputs/predictions_v2.csv')
print(f'V1 accuracy: {v1.correct.mean():.3f}')
print(f'V2 accuracy: {v2.correct.mean():.3f}')
"
```

## Advanced Workflow Options

### Custom Training Parameters
```bash
# Version with specific hyperparameters
python -m audioloop.train training_sets/training_set_v1.csv \
    -v 1_lr001 --learning-rate 0.001 --epochs 500
# → Creates: outputs/model_v1_lr001.pt

python -m audioloop.train training_sets/training_set_v1.csv \
    -v 1_lr0001 --learning-rate 0.0001 --epochs 1000
# → Creates: outputs/model_v1_lr0001.pt
```

### Custom Selection Criteria
```bash
# High-confidence selection
python -m audioloop.active_learning --class-name siren \
    --model outputs/model_v1.pt --run-number 1_highconf \
    --min-confidence 0.95 --total-candidates 10

# Balanced selection
python -m audioloop.active_learning --class-name siren \
    --model outputs/model_v1.pt --run-number 1_balanced \
    --min-confidence 0.8 --positive-pct 0.5 --total-candidates 30
```

### Multi-Class Workflows (Simplified with Experiments)
```bash
# No manual directory creation or file moving needed!
# Each experiment automatically gets its own organized directories

# Siren workflow - automatically organized
python -m audioloop.active_learning --class-name siren --run-number 1 --experiment siren_detection
# → Creates: outputs_siren_detection/predictions_v1.csv, outputs_siren_detection/labeling_candidates_v1.csv

# Dog bark workflow - automatically organized  
python -m audioloop.active_learning --class-name dog_bark --run-number 1 --experiment dog_bark_detection
# → Creates: outputs_dog_bark_detection/predictions_v1.csv, outputs_dog_bark_detection/labeling_candidates_v1.csv

# Gun shot workflow - automatically organized
python -m audioloop.active_learning --class-name gun_shot --run-number 1 --experiment gun_shot_detection
# → Creates: outputs_gun_shot_detection/predictions_v1.csv, outputs_gun_shot_detection/labeling_candidates_v1.csv

# Clean separation, no manual organization needed!
```

## Troubleshooting Version Issues

### Version Mismatch
```bash
# Error: Model outputs/model_v2.pt not found
# Solution: Train the model first
python -m audioloop.train training_sets/training_set_v2.csv

# Error: No training_set_v3.csv after merge
# Solution: Check that merge completed successfully
python -m audioloop.merge_labels training_sets/training_set_v2.csv outputs/labeling_candidates_v2.csv
```

### File Conflicts
```bash
# Error: File already exists
# Solution: Use explicit versioning
python -m audioloop.active_learning --class-name siren --run-number 1b --model outputs/model_v1.pt
```

### Lost Track of Versions
```bash
# List all versions across experiments
find . -name "model_*.pt" | sort -V
find . -name "training_set_*.csv" | sort -V
find . -name "predictions_*.csv" | sort -V

# Find latest version in specific experiment
ls outputs_myexp/model_*.pt | sort -V | tail -n 1

# Overview of all experiments
ls -d outputs_*/ training_sets_*/
```

## Integration with External Tools

### Git Version Control
```bash
# Tag major versions
git add training_sets/training_set_v1.csv outputs/model_v1.pt
git commit -m "Initial model v1 - 95% training accuracy"
git tag v1.0

git add training_sets/training_set_v2.csv outputs/model_v2.pt  
git commit -m "Model v2 - improved with 20 new labels"
git tag v2.0
```

### Experiment Tracking
```bash
# Log version progression
echo "$(date): Created model_v1.pt with 95% accuracy" >> experiment_log.txt
echo "$(date): Model_v2.pt improved to 97% accuracy" >> experiment_log.txt
```

See [README.md](README.md) for project overview and [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed command reference.