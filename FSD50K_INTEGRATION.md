# FSD50K Dataset Integration Guide

This document describes how to add FSD50K dataset support to AudioLoop for binary classification tasks.

## Overview

AudioLoop now supports the FSD50K dataset alongside UrbanSound8K. FSD50K contains 200 classes with multi-label annotations, which we convert to binary classification using two strategies:

1. **One-vs-All**: Any single class vs. everything else
2. **Semantic Grouping**: Multiple related classes vs. everything else

## Dataset Structure

### FSD50K vs UrbanSound8K
| Aspect | UrbanSound8K | FSD50K |
|--------|--------------|---------|
| Classes | 10 | 200 |
| Labels | Single-label | Multi-label |
| Samples | ~8,700 | ~40,000 (dev) + ~10,000 (eval) |
| Organization | 10 folds | train/val split |
| File naming | Descriptive | Numeric IDs |

### Required Files
```
data/FSD50K/
├── FSD50K.dev_audio/          # Audio files (*.wav)
├── FSD50K.ground_truth/
│   ├── vocabulary.csv         # Class definitions
│   ├── dev.csv               # Development set labels
│   └── eval.csv              # Evaluation set labels
└── FSD50K.metadata/
    └── class_info_FSD50K.json # Class descriptions
```

## Implementation

### Core Module: `audioloop/datasets/fsd50k.py`

Key components:
- `FSD50KConfig`: Configuration dataclass
- `FSD50KProcessor`: Main processing class  
- `SEMANTIC_GROUPS`: Predefined class groupings
- Helper functions: `get_class_name()`, `get_class_id()`, `list_classes()`

### Binary Classification Strategies

#### 1. One-vs-All Strategy
```python
from audioloop.datasets.fsd50k import FSD50KConfig, FSD50KProcessor

config = FSD50KConfig()
processor = FSD50KProcessor(config)

# Create binary labels: Piano vs everything else
labels_csv = processor.create_binary_labels_one_vs_all(
    positive_class="Piano",
    split='dev',
    output_csv="outputs/fsd50k_piano_binary.csv"
)
```

#### 2. Semantic Group Strategy
```python
# Use predefined group
labels_csv = processor.create_binary_labels_semantic_group(
    group_name="musical_instruments",
    split='dev',
    output_csv="outputs/fsd50k_music_binary.csv"
)

# Use custom group
string_instruments = {'Guitar', 'Acoustic_guitar', 'Electric_guitar', 'Bass_guitar'}
labels_csv = processor.create_binary_labels_semantic_group(
    group_name="strings",
    positive_classes=string_instruments,
    output_csv="outputs/fsd50k_strings_binary.csv"
)
```

### Predefined Semantic Groups

The module includes 5 predefined semantic groups:

1. **musical_instruments** (23 classes): All musical instruments and music
2. **human_sounds** (9 classes): Speech, conversation, laughter, etc.
3. **animal_sounds** (7 classes): Dogs, cats, birds, etc.
4. **vehicle_sounds** (8 classes): Cars, trucks, aircraft, etc.
5. **mechanical_sounds** (3 classes): Tools, drills, etc.

## Usage Examples

### Basic Usage
```bash
# List all FSD50K classes
python -c "from audioloop.datasets.fsd50k import list_classes; list_classes()"

# List semantic groups
python -c "from audioloop.datasets.fsd50k import list_semantic_groups; list_semantic_groups()"

# Create binary labels for piano detection
python fsd50k_example.py --class Piano --output outputs/piano_labels.csv

# Create binary labels for musical instruments
python fsd50k_example.py --group musical_instruments --output outputs/music_labels.csv
```

### Integration with Existing Workflow

The generated binary labels are compatible with existing AudioLoop scripts:

```bash
# 1. Create binary labels
python fsd50k_example.py --class Guitar --output outputs/fsd50k_guitar.csv

# 2. Generate spectrograms for FSD50K
python -m audioloop.create_all_specs --dataset fsd50k

# 3. Train model (modify simple_train.py to use FSD50K processor)
python -m audioloop.simple_train outputs/fsd50k_guitar.csv

# 4. Run active learning (modify to use FSD50K processor)
python -m audioloop.active_learning --model outputs/model_v1.pt --class-name Guitar

# 5. Label candidates
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv

# 6. Merge labels
python -m audioloop.merge_labels outputs/fsd50k_guitar.csv outputs/labeling_candidates_v1.csv
```

## Binary Label CSV Format

Generated CSV files follow this format:

```csv
filename,label,original_labels,strategy,split
64760,1,"Electric_guitar,Guitar,Plucked_string_instrument,Musical_instrument,Music",one_vs_all_Guitar,train
175151,0,"Sine_wave,Tone",one_vs_all_Guitar,train
```

Fields:
- `filename`: FSD50K file ID
- `label`: Binary label (0/1)
- `original_labels`: Original multi-label classes (comma-separated)
- `strategy`: Binary classification strategy used
- `split`: Original split (train/val from dev.csv)

## Data Statistics

### Class Distribution Examples
- **Piano**: 779 positive / 40,187 negative (1.9% positive)
- **Musical Instruments**: 12,767 positive / 28,199 negative (31.2% positive)  
- **String Instruments**: 3,839 positive / 37,127 negative (9.4% positive)

### Comparison with UrbanSound8K
- FSD50K offers much more diverse classes and larger dataset
- Multi-label nature allows for richer binary classification strategies
- Semantic grouping enables exploration of hierarchical relationships
- Better suited for complex audio understanding tasks

## Integration Steps

To fully integrate FSD50K with existing AudioLoop workflow:

### 1. Modify Training Scripts
Update `simple_train.py` to accept dataset type parameter:
```python
# Add dataset parameter
parser.add_argument('--dataset', choices=['urbansound8k', 'fsd50k'], default='urbansound8k')

# Use appropriate processor based on dataset
if args.dataset == 'fsd50k':
    config = FSD50KConfig()
    processor = FSD50KProcessor(config)
else:
    config = UrbanSound8KConfig()  
    processor = UrbanSound8KProcessor(config)
```

### 2. Modify Active Learning Scripts
Update `active_learning.py` to handle FSD50K class names and binary strategies.

### 3. Spectrogram Generation
Use the updated `create_all_specs.py` script which now supports both datasets:

```bash
# Process UrbanSound8K dataset (default, clears directory first)
python -m audioloop.create_all_specs

# Process FSD50K dataset (clears directory first)
python -m audioloop.create_all_specs --dataset fsd50k

# Process FSD50K eval split
python -m audioloop.create_all_specs --dataset fsd50k --split eval

# Process without clearing existing spectrograms
python -m audioloop.create_all_specs --dataset fsd50k --no-clear
```

The script automatically:
- **Clears** the `data/all_specs` directory before processing (unless `--no-clear` is used)
- **Processes** audio files to spectrograms using the appropriate dataset processor
- **Saves** all spectrograms to the shared `data/all_specs` directory

### 4. Extend Dataset Utilities
Add FSD50K support to `utils/spectrogram_dataset.py` for loading FSD50K spectrograms.

## Testing

Use the provided test scripts to verify functionality:

```bash
# Test basic functionality
python test_fsd50k.py --test basic

# Test binary classification
python test_fsd50k.py --test binary

# Test audio processing (if audio files available)
python test_fsd50k.py --test audio

# Run comprehensive examples
python fsd50k_example.py --compare
python fsd50k_example.py --workflow --class Piano
```

## Benefits for Binary Classification

1. **Larger Dataset**: 4-5x more samples than UrbanSound8K
2. **More Classes**: 200 vs 10 classes enables more diverse binary tasks
3. **Multi-label Nature**: Enables sophisticated binary strategies
4. **Semantic Relationships**: Hierarchical groupings for research
5. **Better Generalization**: More diverse audio content

## Next Steps

1. **Full Integration**: Modify all existing scripts to support both datasets
2. **Cross-Dataset Evaluation**: Train on one dataset, test on another
3. **Hierarchical Classification**: Explore parent-child class relationships
4. **Multi-Binary Models**: Train separate binary classifiers for multiple classes
5. **Dataset Mixing**: Combine UrbanSound8K and FSD50K for larger training sets

## Files Added

- `audioloop/datasets/fsd50k.py` - Core FSD50K dataset processor
- `test_fsd50k.py` - Test suite for FSD50K functionality  
- `fsd50k_example.py` - Usage examples and integration demos
- `FSD50K_INTEGRATION.md` - This documentation file

The implementation follows the same patterns as UrbanSound8K for consistency while adding the flexibility needed for multi-label binary classification.