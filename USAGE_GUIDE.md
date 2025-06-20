# AudioLoop Usage Guide

## Quick Start

AudioLoop is a generalized active learning framework for binary audio classification. You can easily set up active learning cycles for any of the 10 UrbanSound8K classes.

## Basic Usage

### Option 1: Command Line (Easiest)

```bash
# List available sound classes
python -m audioloop.active_learning --list-classes

# Run siren detection with class name
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt

# Run dog bark detection with class name
python -m audioloop.active_learning --class-name dog_bark --model outputs/model_v1.pt

# Run gunshot detection with class ID
python -m audioloop.active_learning --class-id 6 --model outputs/model_v1.pt
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

**CLI (Command Line):**
```bash
# Train on your initial training set
uv run python -m audioloop.simple_train training_sets/training_set_v1.csv

# With custom options
uv run python -m audioloop.simple_train training_sets/training_set_v1.csv \
    --output outputs/model_v1.pt \
    --epochs 500 \
    --learning-rate 0.001
```

**Python API:**
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

**CLI (Command Line):**
```bash
# Run siren detection
uv run python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt

# Run with custom parameters
uv run python -m audioloop.active_learning \
    --class-name dog_bark \
    --model outputs/model_v1.pt \
    --run-number 2 \
    --num-positive 15 \
    --num-negative 5 \
    --min-confidence 0.85

# Run with class ID instead of name
uv run python -m audioloop.active_learning --class-id 3 --model outputs/model_v1.pt

# List all available classes
uv run python -m audioloop.active_learning --list-classes
```

Available options:
- `--class-name`: Specify class by name (e.g., siren, dog_bark)
- `--class-id`: Specify class by ID (0-9)
- `--model`: Path to trained model (required)
- `--run-number`: Version number for output files (default: 1)
- `--negative-name`: Custom name for negative class
- `--num-positive`: Number of positive candidates (default: 10)
- `--num-negative`: Number of negative candidates (default: 10)
- `--min-confidence`: Minimum confidence threshold (default: 0.8)

**Python API:**
```python
from audioloop.active_learning import run_active_learning_for_class

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

AudioLoop includes a simple terminal-based audio labeling tool to streamline this process:

```bash
# Use the interactive labeling tool
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# If audio files are in a different directory
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/audio
```

The tool will:
- Play each audio file automatically
- Show prediction info (filename, confidence, etc.)
- Accept simple keyboard commands: `1` for positive, `0` for negative
- Track your progress and auto-save when you quit

For detailed usage, see `LABELING_GUIDE.md`.

**Manual option:** If you prefer, you can still manually:
1. Open `outputs/labeling_candidates_v1.csv`
2. Review the high-confidence predictions
3. Fill in the `needs_human_label` column with `0` (negative) or `1` (positive)
4. Save the file

### 6. Merge Human Labels
After completing human labeling, merge the labels back into your training set:

```bash
# Merge human labels back into training set
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# This creates training_sets/training_set_v2.csv with combined data
```

Or use the Python API:
```python
from audioloop.active_learning import merge_human_labels

new_training_set = merge_human_labels(
    "training_sets/training_set_v1.csv",
    "outputs/labeling_candidates_v1.csv"  # With human labels filled in
)
```

### 7. Iterate
Repeat steps 3-6 with expanded training sets for each cycle.

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
    predictions_file="outputs/predictions.csv",
    num_positive=15,        # More positive samples
    num_negative=5,         # Fewer negative samples
    min_confidence=0.9,     # Higher confidence threshold
    candidates_csv="outputs/high_conf_candidates.csv"
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
    python -m audioloop.active_learning \
        --class-id $class_id \
        --model outputs/model.pt \
        --run-number 1
done

# Or process specific classes by name
for class_name in siren dog_bark gun_shot; do
    python -m audioloop.active_learning \
        --class-name $class_name \
        --model outputs/model.pt
done
```

## Generated Files

**One-time setup:**
- `data/specs/` - Precomputed spectrograms (default directory)
- `data/all_specs/` - Alternative directory for full UrbanSound8K dataset

**Each active learning cycle generates:**
- `outputs/binary_labels_v1.csv` - Binary labels for the target class
- `outputs/predictions_v1.csv` - Model predictions on full dataset  
- `outputs/labeling_candidates_v1.csv` - High-confidence samples for human review

## Label Management Workflow

### Complete Active Learning Cycle

The complete active learning workflow includes merging human labels back into training sets:

```bash
# 1. Run initial active learning cycle
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt

# 2. Human fills in 'needs_human_label' column in outputs/labeling_candidates_v1.csv

# 3. Merge human labels back into training set
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv

# 4. Train new model with expanded training set (training_sets/training_set_v2.csv)

# 5. Run next cycle with new model
python -m audioloop.active_learning --class-name siren --model outputs/model_v2.pt --run-number 2
```

### Step-by-Step Example

Here's a complete example of running two active learning cycles for siren detection:

```bash
# Initial setup (one-time)
uv run python -m audioloop.create_all_specs

# Cycle 1: Start with initial training set
python -m audioloop.active_learning --class-name siren --model outputs/model_v1.pt --run-number 1
# → Generates outputs/labeling_candidates_v1.csv

# Human reviews outputs/labeling_candidates_v1.csv and fills in needs_human_label column:
# Example content after human labeling:
# filename,predicted_is_positive,confidence,needs_human_label,...
# 24347-8-0-22.wav,1,0.95,1,...  # Human confirms: yes, this is a siren
# 157821-3-0-15.wav,1,0.87,0,...  # Human corrects: no, this is not a siren

# Merge human labels into training set
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv
# → Creates training_sets/training_set_v2.csv with additional samples

# Train model v2 with expanded training set
python -m audioloop.simple_train training_sets/training_set_v2.csv outputs/model_v2.pt

# Cycle 2: Use improved model
python -m audioloop.active_learning --class-name siren --model outputs/model_v2.pt --run-number 2
# → Generates outputs/labeling_candidates_v2.csv with better predictions

# Continue the process...
```

### Manual Label Management

For more control over the labeling process:

```bash
# Merge with custom output location
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv -o training_sets/custom_v2.csv
```

### Python API for Label Management

```python
from audioloop.active_learning import merge_human_labels

# Simple merge with auto-versioning
new_training_set = merge_human_labels(
    "training_sets/training_set_v1.csv",
    "outputs/labeling_candidates_v1.csv"
)
print(f"Created: {new_training_set}")
# Output: Created: training_sets/training_set_v2.csv

# Custom output path
new_training_set = merge_human_labels(
    "training_sets/training_set_v1.csv",
    "outputs/labeling_candidates_v1.csv",
    "training_sets/training_set_experimental.csv"
)

# Full workflow automation
from audioloop.active_learning import run_active_learning_for_class

# Run cycle 1
predictions, candidates = run_active_learning_for_class(
    positive_class_name="siren",
    model_path="outputs/model_v1.pt",
    run_number=1
)

# After human labeling, merge and continue
new_training_set = merge_human_labels(
    "training_sets/training_set_v1.csv", 
    candidates
)

# Continue with cycle 2...
```

## Human Labeling Guidelines

### What to Look For
When reviewing `outputs/labeling_candidates_vX.csv`:

1. **High Confidence Mistakes**: The model is very confident but wrong
   - These are the most valuable for learning
   - Example: Model predicts "siren" with 95% confidence, but it's actually a car horn

2. **Borderline Cases**: Confidence around 0.6-0.8
   - These help define decision boundaries
   - Example: Distant siren that's hard to distinguish from background noise

3. **Representative Samples**: Diverse examples of the target class
   - Different recording conditions, distances, overlapping sounds
   - Helps model generalize better

### Filling in Labels
In the `needs_human_label` column:
- **1**: Positive class (e.g., "yes, this is a siren")  
- **0**: Negative class (e.g., "no, this is not a siren")
- **Leave empty**: Skip if unsure (won't be included in training)

### Quality vs Quantity
- **Better**: 10 high-quality, confident labels
- **Worse**: 50 rushed, uncertain labels
- Take time to listen carefully to each sample

## Automated Workflow Example

For a complete demonstration of the active learning workflow, use the included example script:

```bash
# Run automated workflow with simulated human labeling
python example_workflow.py --class-name siren --cycles 2

# Run with manual human labeling (you'll be prompted to fill in labels)
python example_workflow.py --class-name dog_bark --cycles 3 --no-simulate

# Use custom model
python example_workflow.py --class-name gun_shot --model outputs/custom_model.pt
```

This script demonstrates:
- Running active learning cycles
- Merging human labels back into training sets  
- Handling multiple cycles automatically
- Both simulated and manual human labeling workflows

## Tips

1. **Start Small**: Begin with 10-20 labeled samples per class
2. **High Confidence**: Use confidence ≥ 0.8 for initial candidate selection
3. **Label Quality**: Focus on confident, clear examples rather than edge cases
4. **Iterate Quickly**: Run short cycles (10-20 labels) rather than long ones (100+ labels)
5. **Monitor Progress**: Check if model performance improves after each cycle
6. **Balance**: Ensure roughly equal positive/negative samples in training
7. **Iterate**: Run 3-5 active learning cycles for best results
8. **Validate**: Always test final model on held-out data

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