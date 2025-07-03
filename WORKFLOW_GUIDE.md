# AudioLoop Versioned Workflow Guide

Complete reference for AudioLoop's versioned active learning workflow patterns.

## Automated Workflow (Recommended)

The `automated_workflow.py` script automates the complete active learning cycle, eliminating manual steps and reducing complexity.

### Quick Start
```bash
# Prerequisites (one-time setup)
python -m audioloop.create_all_specs
python -m audioloop.utils.start_labeling --class-name siren --n 40

# Fully automated workflow (auto-labeling for testing)
python automated_workflow.py --class-name siren --cycles 3 --auto-label

# Semi-automated workflow (pause for human labeling)
python automated_workflow.py --class-name dog_bark --cycles 2
```

### What the Automated Workflow Does
1. **Train Model**: Uses existing training set to create `model_v{N}.pt`
2. **Active Learning**: Runs inference to generate `labeling_candidates_v{N}.csv`
3. **Label Candidates**: Either auto-labels or pauses for human labeling
4. **Merge Labels**: Creates expanded `training_set_v{N+1}.csv`
5. **Repeat**: Continues for specified number of cycles

### Advanced Usage
```bash
# Custom training parameters
python automated_workflow.py --class-name gun_shot --cycles 3 --epochs 500 --batch-size 64

# Custom active learning parameters
python automated_workflow.py --class-name jackhammer --cycles 2 --candidates 100 --positive-pct 0.8

# FSD50K dataset
python automated_workflow.py --class-name Drill --cycles 2 --dataset fsd50k --auto-label
```

### Parameters
- `--class-name`: Target class for binary classification
- `--cycles`: Number of active learning cycles to run
- `--auto-label`: Use ground truth for automatic labeling (testing)
- `--dataset`: Dataset choice (`urbansound8k` or `fsd50k`)
- `--epochs`: Max training epochs per cycle
- `--candidates`: Number of candidates to select per cycle
- `--positive-pct`: Target percentage of positive samples

## Manual Workflow (Step-by-step)

For fine-grained control or educational purposes, you can run each step manually:

## Version Naming Convention

AudioLoop uses consistent versioning across all artifacts:
- **Models**: `outputs/model_v{N}.pt`
- **Training sets**: `training_sets/training_set_v{N}.csv`
- **Predictions**: `outputs/predictions_v{N}.csv`
- **Candidates**: `outputs/labeling_candidates_v{N}.csv`
- **Binary labels**: `outputs/binary_labels_v{N}.csv`

## Standard Workflow Pattern

### Complete 3-Cycle Example (Dog Bark Detection)

```bash
# === CYCLE 1: Initial Training ===
# 1. Train initial model (auto-detects v1 from filename)
python -m audioloop.simple_train training_sets/training_set_v1.csv
# → Creates: outputs/model_v1.pt

# 2. Run active learning (creates complete inference record and candidates)
python -m audioloop.active_learning --class-name dog_bark --run-number 1
# → Creates: outputs/predictions_v1.csv, outputs/labeling_candidates_v1.csv

# 3. Human labeling
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k

# 4. Merge labels (auto-creates v2)
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv
# → Creates: training_sets/training_set_v2.csv

# === CYCLE 2: Improved Model ===
# 5. Train improved model
python -m audioloop.simple_train training_sets/training_set_v2.csv
# → Creates: outputs/model_v2.pt

# 6. Run active learning cycle 2
python -m audioloop.active_learning --class-name dog_bark --run-number 2
# → Creates: outputs/predictions_v2.csv, outputs/labeling_candidates_v2.csv

# 7. Human labeling cycle 2
python -m audioloop.label_audio outputs/labeling_candidates_v2.csv --audio-dir data/urbansound8k

# 8. Merge labels for cycle 3
python -m audioloop.merge_labels training_sets/training_set_v2.csv outputs/labeling_candidates_v2.csv
# → Creates: training_sets/training_set_v3.csv

# === CYCLE 3: Final Iteration ===
# 9. Train final model
python -m audioloop.simple_train training_sets/training_set_v3.csv
# → Creates: outputs/model_v3.pt

# 10. Final active learning cycle
python -m audioloop.active_learning --class-name dog_bark --run-number 3
# → Creates: outputs/predictions_v3.csv, outputs/labeling_candidates_v3.csv
```

## Version Auto-Detection

AudioLoop automatically detects versions from filenames:

### Training Models
```bash
# Version auto-detected from training_set_v{N}.csv
python -m audioloop.simple_train training_sets/training_set_v1.csv
# → Creates: outputs/model_v1.pt

python -m audioloop.simple_train training_sets/training_set_v2.csv  
# → Creates: outputs/model_v2.pt

# Override auto-detection
python -m audioloop.simple_train training_sets/training_set_v1.csv -v 5
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
python -m audioloop.simple_train training_sets/training_set_v2.csv -v 2a
python -m audioloop.simple_train training_sets/training_set_v2.csv -v 2b --epochs 1000

# Compare results
python -m audioloop.active_learning --model outputs/model_v2a.pt --run-number 2a
python -m audioloop.active_learning --model outputs/model_v2b.pt --run-number 2b
```

### Pattern 3: Class-Specific Versioning
```bash
# Different version tracks for different classes
python -m audioloop.active_learning --class-name siren --run-number 1
# → outputs/labeling_candidates_v1.csv

python -m audioloop.active_learning --class-name dog_bark --run-number 1  
# → outputs/labeling_candidates_v1.csv (overwrites!)

# Better: Use class-specific naming
python -m audioloop.active_learning --class-name siren --run-number 1
# Manual rename: mv outputs/labeling_candidates_v1.csv outputs/siren_candidates_v1.csv
```

## File Naming Reference

| File Type | Pattern | Auto-Generated | Example |
|-----------|---------|----------------|---------|
| Training Set | `training_sets/training_set_v{N}.csv` | By merge_labels | `training_sets/training_set_v1.csv` |
| Model | `outputs/model_v{N}.pt` | By simple_train | `outputs/model_v1.pt` |
| Predictions | `outputs/predictions_v{N}.csv` | By active_learning | `outputs/predictions_v1.csv` |
| Candidates | `outputs/labeling_candidates_v{N}.csv` | By active_learning | `outputs/labeling_candidates_v1.csv` |
| Binary Labels | `outputs/binary_labels_v{N}.csv` | By active_learning | `outputs/binary_labels_v1.csv` |

## Version Tracking Best Practices

### 1. Keep All Versions
```bash
# Don't delete intermediate versions
ls outputs/
model_v1.pt  model_v2.pt  model_v3.pt  # Keep all for comparison

ls training_sets/
training_set_v1.csv  training_set_v2.csv  training_set_v3.csv  # Track progression
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
python -m audioloop.simple_train training_sets/training_set_v1.csv \
    -v 1_lr001 --learning-rate 0.001 --epochs 500
# → Creates: outputs/model_v1_lr001.pt

python -m audioloop.simple_train training_sets/training_set_v1.csv \
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

### Multi-Class Workflows
```bash
# Track separate workflows for different classes
mkdir -p outputs/siren outputs/dog_bark outputs/gun_shot

# Siren workflow
python -m audioloop.active_learning --class-name siren --run-number 1
mv outputs/predictions_v1.csv outputs/siren/
mv outputs/labeling_candidates_v1.csv outputs/siren/

# Dog bark workflow  
python -m audioloop.active_learning --class-name dog_bark --run-number 1
mv outputs/predictions_v1.csv outputs/dog_bark/
mv outputs/labeling_candidates_v1.csv outputs/dog_bark/
```

## Troubleshooting Version Issues

### Version Mismatch
```bash
# Error: Model outputs/model_v2.pt not found
# Solution: Train the model first
python -m audioloop.simple_train training_sets/training_set_v2.csv

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
# List all versions
ls -la outputs/model_*.pt | sort -V
ls -la training_sets/training_set_*.csv | sort -V  
ls -la outputs/predictions_*.csv | sort -V

# Find latest version
ls outputs/model_*.pt | sort -V | tail -n 1
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