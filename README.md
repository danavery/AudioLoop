# AudioLoop: Active Learning Framework for Audio Classification

AudioLoop is an active learning framework for binary audio classification that minimizes human labeling effort through strategic sample selection. The system implements a versioned workflow for iterative model improvement using human-in-the-loop feedback.

## Quick Start

```bash
# Install dependencies
uv sync

# Generate spectrograms (one-time setup)
python -m audioloop.create_specs

# Create initial training set
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50

# Run automated workflow
python -m audioloop.automated_workflow --class-name Drill --cycles 3 --evaluation-mode --auto-label
```

For detailed usage instructions, see **[USAGE_GUIDE.md](USAGE_GUIDE.md)**.

## Core Features

- **Strategic Sample Selection**: Entropy-based (default), confidence, mixed-entropy, and transition-based active learning strategies
- **Versioned Workflows**: Automatic file versioning for reproducible experiments  
- **Multiple Datasets**: Built-in support for FSD50K and UrbanSound8K
- **Flexible Models**: Pluggable CNN architectures with easy extensibility
- **Human-in-the-Loop**: Terminal and web interfaces for audio labeling
- **Experiment Organization**: Unified configuration system with automatic path management

## How It Works

**Active Learning Concept**: Instead of labeling thousands of audio samples randomly, AudioLoop uses model confidence to strategically select the most informative samples for human review.

**Typical Workflow**:
1. Start with a small labeled dataset (20-50 samples)
2. Train a model on this small set
3. Run inference on unlabeled data  
4. Select high-confidence predictions for human verification
5. Add verified labels to training set and repeat

**Result**: Achieve high model performance with 10-20x fewer labeled samples than traditional approaches.

**Two Modes**:
- **Production Mode**: Deploy on truly unknown audio data
- **Evaluation Mode**: Test and research with datasets having ground truth

## Supported Datasets

**FSD50K** (Primary)
- 200 audio categories, ~50,000 samples
- Multi-label annotations for rich binary classification strategies
- Examples: Piano, Guitar, Speech, Gunshot, Siren

**UrbanSound8K** (Extended)
- 10 urban sound categories, ~8,700 samples  
- Examples: car horn, dog bark, siren, gunshot, drilling

**Custom Datasets**
- Easy extensibility through simple file-based convention
- Automatic discovery - just drop config files in place

## Model Architecture

**Primary**: 5-layer CNN with adaptive pooling for variable-length spectrograms
**Alternative**: Lightweight CNN for resource-constrained environments
**Extensible**: Simple interface for adding custom PyTorch models or HuggingFace models

**Input**: Mel-spectrograms (128 frequency bands, variable time dimension)
**Output**: Binary classification probabilities with confidence scores

## Active Learning Strategies

**High-Confidence Selection** (Default)
- Selects samples where model is most confident
- Effective for early training cycles
- Helps verify model is learning correctly

**Entropy-Based Selection** (Uncertainty Sampling)
- Selects samples where model is most uncertain
- Useful when model becomes overconfident
- Focuses on decision boundary cases

**Basic Transition**
- Automatically switches from confidence to entropy based on model performance
- Adaptive thresholds based on dataset characteristics
- Recommended for most use cases

## Human Labeling Interfaces

**Web UI** (Recommended)
- Modern browser-based interface with audio player
- Visual progress tracking and keyboard shortcuts
- Easy navigation and session resumption

**Terminal Interface**
- Command-line audio labeling tool
- Efficient for experienced users
- Works in any environment with audio support

## Common Use Cases

**Security & Safety**
- Gunshot detection for public safety monitoring
- Siren detection for emergency response systems
- Drilling/construction noise monitoring

**Environmental Monitoring** 
- Wildlife audio monitoring and species detection
- Urban soundscape analysis and noise pollution tracking
- Vehicle sound analysis for traffic management

**Custom Applications**
- Any audio classification task with limited labeled data
- Domain adaptation for new audio environments
- Research on active learning strategies

## Getting Started

1. **Installation**:
   ```bash
   git clone <repository>
   cd audioloop
   uv sync
   ```

2. **Prepare Data**:
   - Download UrbanSound8K or FSD50K dataset
   - Generate spectrograms: `python -m audioloop.create_specs`

3. **Run Example**:
   ```bash
   python -m audioloop.automated_workflow --class-name siren --cycles 2 --evaluation-mode --auto-label
   ```

4. **Explore**: See detailed guides below for specific workflows

## Documentation

**For Users:**
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)**: Complete command reference and examples
- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)**: End-to-end workflow patterns and best practices
- **[LABELING_GUIDE.md](LABELING_GUIDE.md)**: Audio labeling tool usage and best practices

**For Developers:**
- **[DEV_GUIDE.md](DEV_GUIDE.md)**: Architecture, development patterns, and extensibility
- **[docs/adding_new_models.md](docs/adding_new_models.md)**: Guide for integrating custom models
- **[docs/candidate_selection_explained.md](docs/candidate_selection_explained.md)**: Deep dive into selection strategies

**Specialized Topics:**
- **[FSD50K_INTEGRATION.md](FSD50K_INTEGRATION.md)**: FSD50K dataset integration guide
- **[docs/stopping_criteria_guide.md](docs/stopping_criteria_guide.md)**: Training stopping criteria
- **[docs/cycle_stopping_criteria.md](docs/cycle_stopping_criteria.md)**: Active learning cycle stopping strategies
- **[docs/shape_compatibility_and_variable_lengths.md](docs/shape_compatibility_and_variable_lengths.md)**: Variable-length spectrogram support
- **[webui/README.md](webui/README.md)**: Web-based labeling interface

## Performance

- **Sample Efficiency**: Achieve high model performance with 10-20x fewer labeled samples than random sampling
- **Training Speed**: Lightweight CNN architectures enable fast iteration cycles
- **Model Quality**: Typically reach 95-100% accuracy on training sets with strategic sample selection

## Dependencies

- **PyTorch**: Neural network training and inference
- **TorchAudio**: Audio processing and spectrogram generation  
- **NumPy**: Numerical operations
- **SoundFile**: Audio file I/O
- **TQDM**: Progress tracking
- **Ruff**: Code formatting and linting

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Citation

[Add citation information if applicable]