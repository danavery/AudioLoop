# AudioLoop Usage Guide

## Quick Start

AudioLoop is a generalized active learning framework for binary audio classification. You can easily set up active learning cycles for any of the 10 UrbanSound8K classes.

## Basic Usage

### Option 1: Command Line (Easiest)

```bash
# List available sound classes
python -m audioloop.run_active_learning --list-classes

# Run siren detection with class name
python -m audioloop.run_active_learning --class-name siren --model outputs/model_v1.pt

# Run dog bark detection with class name
python -m audioloop.run_active_learning --class-name dog_bark --model outputs/model_v1.pt

# Run gunshot detection with class ID
python -m audioloop.run_active_learning --class-id 6 --model outputs/model_v1.pt
```

### Option 2: Python API

```python
# Simple approach - just provide class name
from audioloop.active_learning import run_active_learning_for_class

predictions_file, candidates_file = run_active_learning_for_class(
    positive_class_name="siren",
    model_path="outputs/model_v1.pt",
    run_number=1
)

# Custom configuration with full control
from audioloop.active_learning import run_active_learning_cycle

predictions_file, candidates_file = run_active_learning_cycle(
    positive_class_id=3,  # dog_bark
    positive_class_name="dog_bark",
    negative_class_name="not_dog_bark", 
    model_path="outputs/model_v1.pt",
    run_number=1
)
```

## Available Sound Classes

| ID | Class Name | Example Use Cases |
|----|------------|-------------------|
| 0 | air_conditioner | HVAC monitoring, indoor environment |
| 1 | car_horn | Traffic monitoring, automotive safety |
| 2 | children_playing | Playground monitoring, residential areas |
| 3 | dog_bark | Pet detection, animal monitoring |
| 4 | drilling | Construction monitoring, noise compliance |
| 5 | engine_idling | Vehicle detection, emissions monitoring |
| 6 | gun_shot | Security systems, public safety |
| 7 | jackhammer | Construction sites, urban planning |
| 8 | siren | Emergency services, traffic management |
| 9 | street_music | Entertainment monitoring, urban soundscapes |

## Easy Class Selection

Just use any UrbanSound8K class name - the system automatically handles binary classification setup:

| Class Name | Use Case |
|------------|----------|
| `siren` | Emergency vehicle detection |
| `dog_bark` | Pet/animal monitoring |
| `gun_shot` | Security applications |
| `car_horn` | Traffic monitoring |
| `drilling` | Construction noise monitoring |
| `air_conditioner` | HVAC monitoring |
| `children_playing` | Playground monitoring |
| `engine_idling` | Vehicle detection |
| `jackhammer` | Construction sites |
| `street_music` | Entertainment monitoring |

## Complete Workflow

### 1. Prepare Data
```bash
# Generate spectrograms for entire UrbanSound8K dataset (one-time setup)
uv run python -m audioloop.create_all_specs
```

### 2. Create Initial Training Set
```bash
# Create training set for siren detection (default)
python -m audioloop.utils.start_labeling

# Or create programmatically for any class
python -c "from audioloop.utils.start_labeling import create_training_set; create_training_set(classname='dog_bark')"
```

```python
from audioloop.utils.start_labeling import create_training_set

# Create training set for any class
create_training_set()  # Default: siren, 10 samples each
create_training_set(classname="dog_bark", n=15)
create_training_set(classname="gun_shot", output_path="training_sets/gunshot_v1.csv")
```

### 3. Train Initial Model
```python
from audioloop.simple_train import run_training

# Train on your initial training set
accuracy = run_training(
    labels_file='training_sets/training_set_v1.csv',
    max_epochs=1000,
    seed=42
)
```

### 4. Run Active Learning Cycle
```python
from audioloop.active_learning import run_active_learning_cycle

# Complete active learning cycle
predictions_file, candidates_file = run_active_learning_for_class(
    positive_class_name="siren",
    model_path="outputs/model_100pct_seed_42.pt",
    run_number=1
)

print(f"Predictions saved to: {predictions_file}")
print(f"Candidates for labeling: {candidates_file}")
```

### 5. Human Labeling
1. Open `outputs/labeling_candidates_cycle1.csv`
2. Review the high-confidence predictions
3. Add human labels to the `needs_human_label` column
4. Add these verified samples to your next training set

### 6. Iterate
Repeat steps 3-5 with expanded training sets for each cycle.

## File Formats

### Training Set CSV
```csv
filepath,label,run
/path/to/audio1.wav,1,1
/path/to/audio2.wav,0,1
```

### Binary Labels CSV (Generated)
```csv
filename,is_positive,original_class,fold
audio1.wav,1,8,3
audio2.wav,0,2,1
```

### Predictions CSV (Generated)
```csv
filename,true_is_positive,predicted_is_positive,prediction,confidence,entropy,prob_negative,prob_positive,correct,original_class,fold,filepath
audio1.wav,1,1,siren,0.95,0.12,0.05,0.95,True,8,3,/path/to/spec.pt
```

## Customization

### Custom Negative Class Names
```python
predictions_file, candidates_file = run_active_learning_for_class(
    positive_class_name="gun_shot",
    negative_class_name="safe_sound",   # Custom negative name
    model_path="outputs/model.pt",
    run_number=1
)
```

### Custom Selection Parameters
```python
from audioloop.active_learning import select_candidates_for_labeling

candidates = select_candidates_for_labeling(
    predictions_csv="outputs/predictions.csv",
    num_positive=15,        # More positive samples
    num_negative=5,         # Fewer negative samples
    min_confidence=0.9,     # Higher confidence threshold
    output_csv="outputs/high_conf_candidates.csv"
)
```

## Common Patterns

### Multi-Class Experiments
```python
# Run multiple binary classification tasks
classes = ["siren", "dog_bark", "gun_shot"]

for class_name in classes:
    predictions, candidates = run_active_learning_for_class(
        positive_class_name=class_name,
        model_path="outputs/model.pt",
        run_number=1
    )
    print(f"Completed {class_name}: {candidates}")
```

### Batch Processing
```bash
# Process multiple classes via command line
for class_id in {0..9}; do
    python -m audioloop.run_active_learning \
        --class-id $class_id \
        --model outputs/model.pt \
        --run-number 1
done
```

## Generated Files

**One-time setup:**
- `data/all_specs/` - Precomputed spectrograms for all 8,732 UrbanSound8K files

**Each active learning cycle generates:**
- `outputs/binary_labels_v1.csv` - Binary labels for the target class
- `outputs/predictions_v1.csv` - Model predictions on full dataset  
- `outputs/labeling_candidates_v1.csv` - High-confidence samples for human review

## Tips

1. **Start Small**: Begin with 10-20 labeled samples per class
2. **High Confidence**: Use confidence ≥ 0.8 for initial candidate selection
3. **Balance**: Ensure roughly equal positive/negative samples in training
4. **Iterate**: Run 3-5 active learning cycles for best results
5. **Validate**: Always test final model on held-out data

## Troubleshooting

### Common Issues

**"Spectrogram file not found"**
- Run `python -m audioloop.create_all_specs` first (one-time setup)
- Check that `data/all_specs/` directory exists with 8,732 .pt files

**"Model file not found"**  
- Train initial model with `simple_train.py`
- Check model path is correct

**Low accuracy on training set**
- Increase learning rate or training epochs
- Check data quality and labels

**No high-confidence candidates**
- Lower confidence threshold (try 0.6-0.7)
- Check model is properly trained

### Getting Help

- `README_agent.md` - Complete framework documentation
- `audioloop/urbansound_classes.py` - Class definitions
- `audioloop/active_learning.py` - Core functions with docstrings