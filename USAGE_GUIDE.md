# AudioLoop Usage Guide

Quick reference for AudioLoop commands and practical usage patterns.

## Command Reference

### Data Preparation
```bash
# Generate spectrograms for entire UrbanSound8K dataset (one-time setup)
python -m audioloop.create_all_specs

# Generate spectrograms for FSD50K dataset
python -m audioloop.create_all_specs --dataset fsd50k
```

### Training Models
```bash
# Train model (version auto-detected from filename)
python -m audioloop.simple_train training_sets/training_set_v1.csv

# Train with explicit version and parameters
python -m audioloop.simple_train training_sets/training_set_v1.csv -v 1 --epochs 500 --batch-size 64
```

### Active Learning Workflow
```bash
# Run active learning cycle (auto-detects version from model)
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt

# Run with explicit parameters
python -m audioloop.active_learning --class-name dog_bark --run-number 2 --total-candidates 20 --positive-pct 0.75 --min-confidence 0.85

# List available sound classes
python -m audioloop.active_learning --list-classes
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
| ID | Name | ID | Name |
|----|------|----|------|
| 0 | air_conditioner | 5 | engine_idling |
| 1 | car_horn | 6 | gun_shot |
| 2 | children_playing | 7 | jackhammer |
| 3 | dog_bark | 8 | siren |
| 4 | drilling | 9 | street_music |

## Common Workflows

### Complete 3-Cycle Example
```bash
# Cycle 1: Initial training
python -m audioloop.simple_train training_sets/training_set_v1.csv
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# Cycle 2: Improved model
python -m audioloop.simple_train training_sets/training_set_v2.csv
python -m audioloop.active_learning --class-name siren --model outputs/model_v2.pt
python -m audioloop.label_audio outputs/labeling_candidates_v2.csv --audio-dir data/urbansound8k
python -m audioloop.merge_labels training_sets/training_set_v2.csv outputs/labeling_candidates_v2.csv

# Cycle 3: Final iteration
python -m audioloop.simple_train training_sets/training_set_v3.csv
python -m audioloop.active_learning --class-name siren --model outputs/model_v3.pt
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
from audioloop.simple_train import run_training

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

# Create for any class
create_training_set(classname="dog_bark", n=15)
create_training_set(classname="gun_shot", output_path="training_sets/gunshot_v1.csv")
```

## Advanced Parameters

### Active Learning Options
```bash
--total-candidates 30           # Number of samples to select (default: 20)
--positive-pct 0.8             # Percentage positive predictions (default: 0.75)
--min-confidence 0.9           # Minimum confidence threshold (default: 0.8)
--negative-name "background"   # Custom negative class name
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

# Model not found
python -m audioloop.simple_train training_sets/training_set_v1.csv  # Train first

# Audio won't play
python -m audioloop.label_audio file.csv --audio-dir /full/path/to/audio

# Version mismatch
python -m audioloop.active_learning --run-number 2 --model outputs/model_v2.pt
```

### Performance Issues
- **Slow training**: Reduce batch size or use CPU with `--device cpu`
- **Out of memory**: Lower batch size or use smaller model
- **Slow audio loading**: Ensure audio files are local, not networked

## Integration Examples

### Custom Workflows
```python
# Multi-class experiment
classes = ["siren", "dog_bark", "gun_shot"]
results = {}

for class_name in classes:
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