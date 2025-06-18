# AudioLoop: Active Learning Framework for Audio Classification

## Overview

AudioLoop is a human-in-the-loop (HITL) active learning system designed for audio classification tasks. The project demonstrates how to efficiently train audio classification models by strategically selecting the most informative samples for human labeling, minimizing the total annotation effort required.

The framework is **generalized for any binary classification task** using the UrbanSound8K dataset. The system can be configured for any of the 10 UrbanSound8K classes (siren, car horn, dog bark, gunshot, etc.) or custom binary classification tasks using simple configuration options.

## Core Concept

Traditional supervised learning requires large labeled datasets. Active learning reduces this burden by:

1. **Starting small**: Begin with a minimal labeled training set (e.g., 20 samples)
2. **Training iteratively**: Train a model, run inference on unlabeled data
3. **Smart selection**: Choose the most informative samples for human labeling
4. **Human feedback**: Get labels for these carefully selected samples
5. **Repeat**: Add new labels to training set and iterate

## Project Structure

```
audioloop/
├── audioloop/                    # Main package
│   ├── active_learning.py        # Generalized active learning workflow (any binary classification)
│   ├── simple_train.py          # Model training logic
│   ├── test_model.py            # Model validation on training data
│   ├── run_active_learning.py   # Command-line interface for active learning
│   ├── urbansound_classes.py    # UrbanSound8K class definitions
│   ├── create_all_specs.py      # Spectrogram generation for full dataset
│   ├── inference.py             # General inference utilities
│   ├── models/                  # Model architectures
│   │   ├── simple_cnn.py        # Lightweight 2-layer CNN (SimpleCNN)
│   │   └── cnn_5layer.py        # 5-layer CNN architecture (SoundCNN)
│   └── utils/                   # Supporting utilities
│       ├── spec_dataset.py      # Dataset for precomputed spectrograms
│       ├── urbansound_dataset.py # UrbanSound8K dataset handler
│       ├── labeled_dataset.py   # Generic labeled dataset
│       └── ...
├── data/
│   ├── urbansound8k/            # UrbanSound8K audio files and metadata
│   │   ├── fold1/, fold2/, ...  # Audio files organized by fold
│   │   └── UrbanSound8K.csv     # Original metadata
│   └── all_specs/               # Precomputed mel-spectrograms (.pt files)
├── outputs/                     # Generated files and results
├── training_sets/               # Versioned training sets
│   ├── training_set_v1.csv      # Initial seed training set
│   ├── training_set_v2.csv      # After first labeling cycle
│   └── ...
├── README_agent.md             # This file
└── USAGE_GUIDE.md              # Quick start and usage examples
```

## Data Flow

### 1. Audio Preprocessing
- Raw audio files → Mel-spectrograms → PyTorch tensors (.pt files)
- Spectrograms: 128 mel bands, ~993 time frames, stereo → mono conversion
- File: `create_all_specs.py` (processes entire UrbanSound8K dataset)

### 2. Training Data Format
CSV format: `filepath,label,run`
```csv
/path/to/audio.wav,1,1
/path/to/audio2.wav,0,1
```
- `filepath`: Full path to audio file
- `label`: Binary label (1=positive class, 0=negative class)
- `run`: Training iteration number

### 3. Binary Labels Format
CSV format: `filename,is_positive,original_class,fold`
```csv
audio1.wav,1,8,3
audio2.wav,0,2,1
```
- `filename`: Audio filename (without path)
- `is_positive`: Binary label (1=positive class, 0=negative class)
- `original_class`: Original UrbanSound8K class ID (0-9)
- `fold`: UrbanSound8K fold number

### 4. Model Architecture
- **SimpleCNN**: Lightweight 2-layer CNN with global average pooling
- Input: Mono spectrograms (1, 128, 993)
- Output: Binary classification logits
- ~4,866 parameters

### 5. Active Learning Cycle

```python
# Using the active learning framework
from audioloop.active_learning import run_active_learning_cycle

# Step 1-3: Complete cycle for any binary classification task
# Simple approach - just provide class name
from audioloop.active_learning import run_active_learning_for_class

predictions_file, candidates_file = run_active_learning_for_class(
    positive_class_name="siren",       # Any UrbanSound8K class name
    model_path="outputs/model_v1.pt",
    run_number=1
)

# Or use the full function with more control
from audioloop.active_learning import run_active_learning_cycle

predictions_file, candidates_file = run_active_learning_cycle(
    positive_class_id=8,               # UrbanSound8K class ID (8=siren)
    positive_class_name="siren",       # Human-readable positive class name
    negative_class_name="not_siren",   # Human-readable negative class name
    model_path="outputs/model_v1.pt",
    run_number=1
)
```

### 6. Command Line Interface

```bash
# Run siren detection with class name
python -m audioloop.run_active_learning --class-name siren --model outputs/model_v1.pt

# Run dog bark detection with class ID
python -m audioloop.run_active_learning --class-id 3 --model outputs/model_v1.pt

# List all UrbanSound8K classes
python -m audioloop.run_active_learning --list-classes
```

## Key Files and Scripts

### Core Workflow
- **`active_learning.py`**: Generalized active learning pipeline for any binary classification
- **`run_active_learning.py`**: Command-line interface for running cycles
- **`urbansound_classes.py`**: Class definitions and utility functions
- **`simple_train.py`**: Training loop with early stopping at 100% accuracy
- **`test_model.py`**: Validates model performance on training data

### Data Handling
- **`utils/start_labeling.py`**: Creates initial training sets for any class
- **`utils/spec_dataset.py`**: Loads precomputed spectrograms for training
- **`utils/urbansound_dataset.py`**: Handles UrbanSound8K format for inference
- **`create_all_specs.py`**: Converts all UrbanSound8K audio files → spectrograms

### Model Architectures
- **`models/cnn_5layer.py`**: Original 5-layer CNN (SoundCNN)
- **`models/simple_cnn.py`**: SimpleCNN - lightweight 2-layer version for binary classification

## Selection Strategies

### Current Implementation: High-Confidence Sampling
- Selects samples where model is most confident (confidence ≥ 0.8)
- Balances positive/negative predictions for any binary classification task
- Excludes already-labeled samples
- Generalizes across all UrbanSound8K classes

### Future Extensions
- **Uncertainty Sampling**: Low-confidence samples near decision boundary
- **Entropy-based**: High-entropy (confused) predictions
- **Diversity Sampling**: Ensures feature space coverage
- **Hybrid Approaches**: Combines multiple strategies

## Typical Workflow

### First Cycle
1. **Prepare data**: `uv run python -m audioloop.create_all_specs`
2. **Create training set**: `python -m audioloop.utils.start_labeling` or `create_training_set(classname="dog_bark")`
3. **Train initial model**: Use `simple_train.py` with `training_set_v1.csv`
4. **Validate**: Run `test_model.py` to verify training worked
5. **Get candidates**: Run `active_learning.py` to generate labeling candidates
6. **Human labeling**: Review `outputs/cycle1_labeling_candidates.csv`

### Subsequent Cycles
1. **Update training set**: Add human labels to `training_set_v2.csv`
2. **Retrain**: Train on expanded dataset
3. **Iterate**: Repeat candidate selection and labeling

## Performance Metrics

### Model Performance
- **Training Accuracy**: Should reach 95-100% on training set
- **Validation Accuracy**: Performance on held-out data
- **Confidence Distribution**: High confidence indicates good uncertainty estimation

### Active Learning Efficiency
- **Labeling Effort**: Number of samples requiring human annotation
- **Learning Curve**: Accuracy vs. number of labeled samples
- **Selection Quality**: How informative are the selected samples?

## Configuration and Paths

### Key Directories
- Spectrograms: `data/all_specs/` (not `data/specs/`)
- Models: `outputs/model_*.pt`
- Predictions: `outputs/*_predictions.csv`
- Candidates: `outputs/*_labeling_candidates.csv`

### Common Gotchas
- **Specs directory**: All spectrograms are stored in `data/all_specs/`
- **File paths**: Training CSV uses full paths, but datasets need just filenames
- **Device**: Auto-detects MPS (Apple Silicon), CUDA, or CPU

## Extension Points

### New Audio Domains
1. Replace UrbanSound8K with domain-specific dataset
2. Adjust spectrogram parameters for audio characteristics
3. Modify class mappings in `urbansound_classes.py`
4. Update binary label creation logic in `active_learning.py`

### Multi-class Classification
1. Update model output layer (`num_classes > 2`)
2. Modify loss function and metrics
3. Extend `active_learning.py` to support multi-class
4. Create new class definition modules for multi-class tasks

### Advanced Selection Strategies
1. Implement uncertainty-based sampling in `active_learning.py`
2. Add clustering for diversity sampling
3. Experiment with acquisition functions
4. Support mixed strategies (high-confidence + uncertainty sampling)

## Example Results

### Initial State (training_set_v1.csv)
- **Training samples**: 20 (10 positive, 10 negative)
- **Training accuracy**: 95% (19/20 correct)
- **Full dataset inference**: 49.1% accuracy, 86.85% confidence

### Active Learning Output
- **High-confidence candidates**: 20 samples (10 pos, 10 neg)
- **Confidence range**: 100% (model is very confident)
- **Ready for human review**: `outputs/labeling_candidates_cycle1.csv`

This demonstrates the classic active learning scenario: the model is confident but often wrong, making human feedback on confident mistakes highly valuable for learning.

### Available Binary Classification Tasks
- **Siren Detection**: Emergency vehicle sirens vs all other sounds
- **Dog Bark Detection**: Dog barks vs all other sounds
- **Gunshot Detection**: Gunshots vs all other sounds
- **Car Horn Detection**: Car horns vs all other sounds
- **Drilling Detection**: Construction drilling vs all other sounds
- **Custom Tasks**: Any UrbanSound8K class vs others

## Dependencies

- **PyTorch**: Neural networks and tensor operations
- **torchaudio**: Audio processing and spectrogram generation
- **tqdm**: Progress bars for long-running operations
- **CSV**: Data manipulation and file I/O

## Future Directions

1. **Multi-domain validation**: Test on datasets beyond UrbanSound8K
2. **Advanced architectures**: Transformer-based audio models
3. **Uncertainty quantification**: Better confidence estimation
4. **Distributed labeling**: Support for multiple human annotators
5. **Real-time inference**: Streaming audio classification

This framework provides a solid foundation for exploring active learning in audio domains with full generalization across binary classification tasks. The system offers easy configuration for any UrbanSound8K class through simple class name specification.
