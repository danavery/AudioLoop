# AudioLoop: Active Learning Framework for Audio Classification

AudioLoop is an active learning framework for binary audio classification that minimizes human labeling effort through strategic sample selection. The system implements a versioned workflow for iterative model improvement using human-in-the-loop feedback.

## Quick Start

### Automated Workflow (Recommended)
```bash
# Install dependencies
uv sync

# Generate spectrograms (one-time setup)
python -m audioloop.create_all_specs

# Create initial training set
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50

# Run fully automated workflow (auto-labeling for testing)
python automated_workflow.py --class-name Drill --cycles 3 --auto-label

# Run semi-automated workflow (pause for human labeling)
python automated_workflow.py --class-name Drill --cycles 2
```

### Manual Workflow (Step-by-step)
```bash
# Install dependencies
uv sync

# Generate spectrograms (one-time setup)
python -m audioloop.create_all_specs

# Train initial model
python -m audioloop.train training_sets/training_set_v1.csv

# Run active learning cycle (production mode - default)
python -m audioloop.active_learning --class-name Drill --run-number 1 --selection-mode basic_transition --auto-thresholds

# For evaluation/research with ground truth, add --with-ground-truth flag
# python -m audioloop.active_learning --class-name Drill --run-number 1 --with-ground-truth

# Label audio samples interactively
python -m audioloop.label_audio outputs/labeling_candidates_v1.csv --audio-dir data/FSD50K/FSD50K.dev_audio

# Merge human labels back into training set
python -m audioloop.merge_labels training_sets/training_set_v1.csv outputs/labeling_candidates_v1.csv
```

## Configuration System

AudioLoop uses a unified configuration system that eliminates hardcoded paths and provides consistent experiment organization:

```python
from audioloop.config import AudioLoopConfig

# Create configuration for an experiment
config = AudioLoopConfig(experiment_name="my_experiment", dataset="urbansound8k")

# All paths are automatically organized
config.output_dir          # outputs_my_experiment/
config.training_sets_dir   # training_sets_my_experiment/
config.specs_dir          # data/all_specs/

# Generate versioned file paths
config.get_model_path(1)        # outputs_my_experiment/model_v1.pt
config.get_predictions_path(1)  # outputs_my_experiment/predictions_v1.csv
```

### Environment Variables
Customize behavior without code changes:
```bash
# Set default dataset
export AUDIOLOOP_DATASET=urbansound8k

# Customize data locations
export AUDIOLOOP_DATA_ROOT=/custom/data
export AUDIOLOOP_OUTPUT_ROOT=/custom/outputs

# Configuration precedence: explicit parameters > env vars > defaults
```

## Core Concept

Traditional supervised learning requires large labeled datasets. AudioLoop reduces this burden through active learning, supporting both production deployment and research workflows:

### Production Mode (Default)
1. **Start small**: Begin with minimal labeled training set (20 samples)
2. **Train iteratively**: Train model, run inference on unlabeled data
3. **Smart selection**: Choose most informative samples using model confidence
4. **Human feedback**: Get labels for carefully selected samples via manual review
5. **Repeat**: Add new labels to training set and iterate
6. **Monitor**: Track prediction confidence and distribution trends

### Evaluation Mode (Research/Testing)
1. **Start small**: Begin with minimal labeled training set from known dataset
2. **Train iteratively**: Train model, run inference with ground truth comparison
3. **Smart selection**: Choose samples using confidence or uncertainty strategies
4. **Auto-labeling**: Automatically apply ground truth labels for rapid testing
5. **Repeat**: Add new labels to training set and iterate
6. **Evaluate**: Track comprehensive metrics (F1, precision, recall, accuracy)

## Project Structure

```
audioloop/
├── audioloop/                    # Main package
│   ├── config.py                # Unified configuration system
│   ├── active_learning.py        # CLI interface for active learning
│   ├── active_learning_core.py   # Core active learning pipeline
│   ├── train.py                 # CLI interface for model training
│   ├── training_core.py         # Core training logic with versioning
│   ├── label_audio.py           # Terminal-based audio labeling
│   ├── merge_labels.py          # Label management utilities
│   ├── create_all_specs.py      # Audio preprocessing
│   ├── models/                  # Neural network architectures
│   │   ├── cnn_5layer.py        # Primary CNN with adaptive pooling
│   │   └── simple_cnn.py        # Lightweight alternative
│   ├── utils/                   # Supporting utilities
│   │   ├── paths.py             # Path utilities and environment config
│   │   ├── spectrogram_dataset.py # Unified dataset loader
│   │   ├── data_utils.py        # Core utilities
│   │   └── log_normalize.py     # Spectrogram normalization
│   └── datasets/                # Dataset handlers
│       ├── dataset_config.py    # Abstract dataset interface
│       ├── urbansound8k.py      # UrbanSound8K integration
│       └── fsd50k.py           # FSD50K integration
├── data/
│   ├── urbansound8k/            # Audio files organized by fold
│   ├── all_specs/               # Precomputed spectrograms
│   └── FSD50K/                  # FSD50K dataset (optional)
├── outputs/                     # Generated models and results
├── training_sets/               # Versioned training datasets
└── docs/                        # Additional documentation
```

## Supported Datasets

### FSD50K (Primary)
- **Classes**: 200 audio categories with multi-label annotations
- **Samples**: ~50,000 audio files
- **Binary Strategies**: One-vs-all or semantic grouping
- **Examples**: musical instruments, human sounds, vehicle sounds

### UrbanSound8K (Extended)
- **Classes**: 10 urban sound categories
- **Samples**: ~8,700 audio files
- **Binary Tasks**: Any class vs. all others
- **Examples**: siren detection, dog bark detection, gunshot detection

## Architecture Overview

### Versioned Workflow System
AudioLoop uses consistent versioning across all artifacts:
- Models: `outputs/model_v{N}.pt`
- Training sets: `training_sets/training_set_v{N}.csv`
- Predictions: `outputs/predictions_v{N}.csv`
- Candidates: `outputs/labeling_candidates_v{N}.csv`

### Neural Network Architecture
- **Primary Model**: 5-layer CNN with adaptive pooling (`cnn_5layer.py`)
- **Alternative**: Lightweight 2-layer CNN (`simple_cnn.py`)
- **Input**: Mel-spectrograms (128 bands, ~993 frames)
- **Output**: Binary classification probabilities

### Data Pipeline
1. **Audio → Spectrograms**: Raw audio converted to mel-spectrograms
2. **Training**: CNN trained on labeled spectrograms
3. **Inference**: Model predictions on unlabeled data
4. **Selection**: High-confidence samples chosen for human review
5. **Labeling**: Human feedback via interactive tool
6. **Integration**: New labels merged into training set

## Available Sound Classes

### UrbanSound8K Classes
| ID | Class Name | Description |
|----|------------|-------------|
| 0 | air_conditioner | HVAC systems, cooling units |
| 1 | car_horn | Vehicle horns, honking |
| 2 | children_playing | Playground sounds, kids |
| 3 | dog_bark | Dog barks and vocalizations |
| 4 | drilling | Construction drilling, tools |
| 5 | engine_idling | Vehicle engines at idle |
| 6 | gun_shot | Gunshots, firearms |
| 7 | jackhammer | Heavy construction equipment |
| 8 | siren | Emergency vehicle sirens |
| 9 | street_music | Street performers, music |

### Usage Examples
```bash
# List all available classes
python -m audioloop.active_learning --list-classes

# Run detection for specific class
python -m audioloop.active_learning --class-name Drill --model outputs/model_v1.pt
python -m audioloop.active_learning --class-name Speech --model outputs/model_v1.pt
```

## File Formats

### Training Set CSV
```csv
filepath,label,run
data/all_specs/100032-3-0-0.pt,1,1
data/all_specs/100263-2-0-117.pt,0,1
```

### Predictions CSV (Generated)
Production Mode (default):
```csv
filename,prediction,predicted_class,confidence,entropy,prob_negative,prob_positive,original_class,fold,filepath
```

Evaluation Mode (with --with-ground-truth):
```csv
filename,ground_truth,prediction,predicted_class,confidence,entropy,prob_negative,prob_positive,correct,original_class,fold,filepath
```

### Candidates CSV (For Human Labeling)
```csv
filename,prediction,predicted_class,confidence,needs_human_label,entropy,prob_negative,prob_positive,original_class,fold,filepath
```

## Key Features

### Active Learning Strategies
- **High-Confidence Sampling**: Selects samples where model is most confident
- **Basic Transition**: Automatically switches from confidence to entropy based on F1, confidence, and variance metrics
- **Adaptive Thresholds**: Auto-calculates optimal thresholds based on class imbalance without requiring ground truth
- **Balanced Selection**: Maintains positive/negative ratio
- **Exclusion Logic**: Avoids re-labeling existing samples

### Human-in-the-Loop Tools
- **Interactive Audio Player**: Terminal-based labeling interface
- **Progress Tracking**: Visual progress bars and session resumption
- **Quality Control**: Skip unclear samples, consistent labeling

### Version Management
- **Auto-Detection**: Versions extracted from filenames
- **Consistent Naming**: Standardized file naming conventions
- **Reproducibility**: All artifacts versioned for experiment tracking

## Performance & Efficiency

### Model Performance
- **Training Speed**: Lightweight CNN architecture for fast iteration
- **Accuracy**: Typically 95-100% on training set
- **Confidence**: High-quality uncertainty estimation for selection

### Active Learning Efficiency
- **Sample Reduction**: 10-20x fewer labels needed vs. random sampling
- **Quality Focus**: Strategic selection of informative samples
- **Iterative Improvement**: Each cycle improves model performance

## Common Use Cases

### Security & Safety
- **Gunshot Detection**: Security systems, public safety monitoring
- **Siren Detection**: Emergency response, traffic management
- **Drilling Detection**: Construction monitoring, noise compliance

### Environmental Monitoring
- **Vehicle Sounds**: Traffic analysis, emissions monitoring
- **Animal Sounds**: Wildlife monitoring, pet detection
- **Urban Soundscapes**: City planning, noise pollution analysis

### Custom Applications
- **Binary Classification**: Any UrbanSound8K class vs. others
- **Multi-Dataset**: Combine UrbanSound8K and FSD50K
- **Domain Adaptation**: Extend to new audio domains

## Development & Extension

### Adding New Datasets
1. Create dataset processor in `audioloop/datasets/`
2. Implement spectrogram generation
3. Update active learning pipeline
4. Add dataset-specific configuration

### Custom Model Architectures
1. Add model definition to `audioloop/models/`
2. Update training script imports
3. Maintain binary classification interface
4. Test with active learning pipeline

### Advanced Selection Strategies
- **Basic Transition**: Switches from confidence to entropy based on performance metrics
- **Adaptive Thresholds**: Automatically adjusts transition criteria for imbalanced datasets
- **Uncertainty Sampling**: Low-confidence samples near decision boundary
- **Entropy-Based**: High-entropy (uncertain) predictions
- **Diversity Sampling**: Ensures feature space coverage
- **Hybrid Approaches**: Combine multiple strategies

## Extensibility

AudioLoop is designed to be easily extensible with custom datasets and models:

### Adding Custom Datasets
```bash
# Copy template to create your dataset
cp audioloop/datasets/templates/simple_audio_template.py audioloop/datasets/my_dataset_config.py

# Edit the file and use immediately
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-classes
```

### Adding Custom Models
```python
# Create audioloop/models/my_model.py
class MyModel(AudioLoopModel):
    # Implement required methods
    
# Use immediately
python -m audioloop.train training_set.csv --model-type my_model
```

Both systems use dynamic discovery - no registration required, just drop files in place and they're automatically available throughout AudioLoop.

## Dependencies

- **PyTorch**: Neural network training and inference
- **TorchAudio**: Audio processing and spectrogram generation
- **NumPy**: Numerical operations and data manipulation
- **SoundFile**: Audio file I/O operations
- **TQDM**: Progress bars for long-running operations
- **Ruff**: Code formatting and linting

## Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Detailed command reference and examples
- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)**: Versioned workflow patterns
- **[LABELING_GUIDE.md](LABELING_GUIDE.md)**: Audio labeling tool usage
- **[FSD50K_INTEGRATION.md](FSD50K_INTEGRATION.md)**: FSD50K dataset integration
- **[Adding New Models Guide](docs/adding_new_models.md)**: How to integrate custom or HuggingFace models
- **[CLAUDE.md](CLAUDE.md)**: AI assistant guidance

## Getting Started

1. **Setup Environment**:
   ```bash
   git clone <repository>
   cd audioloop
   uv sync
   ```

2. **Prepare Data**:
   ```bash
   # Download UrbanSound8K dataset to data/urbansound8k/
   python -m audioloop.create_all_specs
   ```

3. **Run Example Workflow**:
   ```bash
   python example_workflow.py --class-name siren --cycles 2
   ```

4. **Explore Documentation**: See individual guides for detailed usage

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Citation

[Add citation information if applicable]