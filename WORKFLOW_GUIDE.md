# AudioLoop Workflow Guide

This guide explains the consistent versioned workflow for active learning with AudioLoop.

## Overview

AudioLoop uses a versioned naming convention throughout the workflow:
- Models: `model_v1.pt`, `model_v2.pt`, etc.
- Training sets: `training_set_v1.csv`, `training_set_v2.csv`, etc.
- Predictions: `predictions_v1.csv`, `predictions_v2.csv`, etc.
- Candidates: `labeling_candidates_v1.csv`, `labeling_candidates_v2.csv`, etc.

## Complete Workflow Example

Here's a complete example for detecting "dog_bark" sounds:

### Step 1: Prepare Initial Training Set

Create your initial training set with binary labels:
```bash
# Create binary labels from UrbanSound8K (dog_bark = class 3)
python -m audioloop.active_learning create-binary-labels \
    --class-id 3 \
    --output training_sets/training_set_v1.csv
```

### Step 2: Train Initial Model

Train your first model (version is auto-detected from filename):
```bash
python -m audioloop.simple_train training_sets/training_set_v1.csv
# Auto-detects version 1 from filename
# Creates: outputs/model_v1.pt
```

### Step 3: Run Active Learning Cycle

Generate predictions and select candidates for labeling:
```bash
python -m audioloop.active_learning --class-name dog_bark --run-number 1
# Automatically uses: outputs/model_v1.pt
# Creates: outputs/predictions_v1.csv
# Creates: outputs/labeling_candidates_v1.csv
```

### Step 4: Label Candidates

Label the selected audio samples:
```bash
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv \
    --audio-dir data/urbansound8k
```

### Step 5: Merge Labels

Merge the human-labeled data back into your training set:
```bash
python -m audioloop.merge_labels \
    training_sets/training_set_v1.csv \
    outputs/labeling_candidates_v1.csv
# Creates: training_sets/training_set_v2.csv
```

### Step 6: Train Next Model

Train the improved model:
```bash
python -m audioloop.simple_train training_sets/training_set_v2.csv
# Auto-detects version 2 from filename
# Creates: outputs/model_v2.pt
```

### Step 7: Continue the Cycle

Run the next active learning cycle:
```bash
python -m audioloop.active_learning --class-name dog_bark --model outputs/model_v2.pt
# Auto-detects run number 2 from model filename
# Creates: outputs/predictions_v2.csv
# Creates: outputs/labeling_candidates_v2.csv
```

## Key Commands Reference

### Training Models
```bash
# Train model (version auto-detected from training_set_v{N}.csv)
python -m audioloop.simple_train training_sets/training_set_v{N}.csv

# Explicitly specify version (overrides auto-detection)
python -m audioloop.simple_train training_sets/training_set_v{N}.csv -v {N}

# Train with custom parameters
python -m audioloop.simple_train training_sets/training_set_v1.csv \
    --epochs 500 \
    --batch-size 64 \
    --learning-rate 0.0005
```

### Active Learning
```bash
# Run active learning cycle (auto-detects version from model filename)
python -m audioloop.active_learning --class-name {class} --model outputs/model_v2.pt
# Auto-detects run number 2, creates predictions_v2.csv and labeling_candidates_v2.csv

# Or specify run-number explicitly (automatically uses outputs/model_v{N}.pt)
python -m audioloop.active_learning --class-name {class} --run-number {N}

# With custom model path (version not auto-detected from non-standard names)
python -m audioloop.active_learning --class-name {class} --run-number {N} \
    --model path/to/custom_model.pt
```

### Labeling Audio
```bash
# Label candidates from active learning
python -m audioloop.label_audio outputs/labeling_candidates_v{N}.csv \
    --audio-dir data/urbansound8k
```

### Merging Labels
```bash
# Merge to create next version of training set
python -m audioloop.merge_labels \
    training_sets/training_set_v{N}.csv \
    outputs/labeling_candidates_v{N}.csv
# Creates: training_sets/training_set_v{N+1}.csv
```

## File Naming Convention

| File Type | Pattern | Example |
|-----------|---------|---------|
| Model | `outputs/model_v{N}.pt` | `outputs/model_v1.pt` |
| Training Set | `training_sets/training_set_v{N}.csv` | `training_sets/training_set_v1.csv` |
| Predictions | `outputs/predictions_v{N}.csv` | `outputs/predictions_v1.csv` |
| Candidates | `outputs/labeling_candidates_v{N}.csv` | `outputs/labeling_candidates_v1.csv` |
| Binary Labels | `outputs/binary_labels_v{N}.csv` | `outputs/binary_labels_v1.csv` |

## Tips for Effective Active Learning

1. **Start Small**: Begin with a small, high-quality training set
2. **Label Consistently**: When labeling, be consistent about edge cases
3. **Monitor Progress**: Track model accuracy across versions
4. **Focus on Errors**: The active learning selection prioritizes uncertain cases
5. **Save Everything**: Keep all versioned files for reproducibility

## Troubleshooting

### Model Not Found
If you see "Model outputs/model_v2.pt not found", make sure you've trained it:
```bash
python -m audioloop.simple_train training_sets/training_set_v2.csv -v 2
```

### Audio Not Playing
Ensure your `--audio-dir` points to the parent of the fold directories:
```bash
# Correct
--audio-dir data/urbansound8k

# Incorrect
--audio-dir data/urbansound8k/fold1
```

### Wrong Predictions File
The predictions file version matches the run number:
- `--run-number 1` creates `predictions_v1.csv`
- `--run-number 2` creates `predictions_v2.csv`

## Example: Complete 3-Cycle Workflow

```bash
# Initial setup
python -m audioloop.active_learning create-binary-labels --class-id 8 \
    --output training_sets/training_set_v1.csv

# Cycle 1
python -m audioloop.simple_train training_sets/training_set_v1.csv  # Creates model_v1.pt
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt  # Auto-detects v1
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Cycle 2
python -m audioloop.simple_train training_sets/training_set_v2.csv  # Creates model_v2.pt
python -m audioloop.active_learning --class-name siren --model outputs/model_v2.pt  # Auto-detects v2
python -m audioloop.label_audio outputs/labeling_candidates_v2.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v2.csv outputs/labeling_candidates_v2.csv

# Cycle 3
python -m audioloop.simple_train training_sets/training_set_v3.csv  # Creates model_v3.pt
python -m audioloop.active_learning --class-name siren --model outputs/model_v3.pt  # Auto-detects v3
python -m audioloop.label_audio outputs/labeling_candidates_v3.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v3.csv outputs/labeling_candidates_v3.csv
```

## Next Steps

- Review model performance metrics in the predictions files
- Analyze which samples are being selected for labeling
- Consider adjusting selection criteria (confidence thresholds, number of samples)
- Export your final model for production use