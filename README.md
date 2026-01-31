# AudioLoop: Active Learning Framework for Audio Classification

AudioLoop is an active learning framework for binary audio classification that minimizes human labeling effort through strategic sample selection. The system implements a versioned workflow for iterative model improvement using human-in-the-loop feedback.

## Quick Start

AudioLoop is installed once and used from separate project directories for each classification task.

### 1. Install AudioLoop (One-Time)

```bash
# Clone to a permanent location
git clone <repository> ~/tools/audioloop
cd ~/tools/audioloop

# Install with uv (recommended)
uv sync

# Optional: Add alias to ~/.bashrc or ~/.zshrc for easy activation
echo 'alias al="source ~/tools/audioloop/.venv/bin/activate"' >> ~/.bashrc
```

### 2. Create a Project (Per Classification Task)

```bash
# Activate audioloop (or use your alias: al)
source ~/tools/audioloop/.venv/bin/activate

# Create and initialize a new project directory
mkdir ~/projects/my-audio-classifier
cd ~/projects/my-audio-classifier
python -m audioloop.init_project

# Download dataset (FSD50K or UrbanSound8K) into data/
# Then generate spectrograms
python -m audioloop.create_specs

# Create initial training set and run
python -m audioloop.utils.create_bootstrap_set --class-name Drill --n 50
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

AudioLoop requires Python 3.11+ and is installed once, then used from separate project directories.

### Step 1: Install AudioLoop (One-Time)

Choose a permanent location for the audioloop installation:

**With uv (recommended):**
```bash
git clone <repository> ~/tools/audioloop
cd ~/tools/audioloop
uv python install 3.11  # Optional: install Python 3.11 if needed
uv sync
```

**With pip:**
```bash
git clone <repository> ~/tools/audioloop
cd ~/tools/audioloop
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Recommended:** Add a shell alias for easy activation:
```bash
# Add to ~/.bashrc or ~/.zshrc
alias al='source ~/tools/audioloop/.venv/bin/activate'
```

### Step 2: Create a Project Directory

Each classification task gets its own directory. Activate audioloop first, then initialize:

```bash
source ~/tools/audioloop/.venv/bin/activate  # Or just: al

mkdir ~/projects/siren-detector
cd ~/projects/siren-detector
python -m audioloop.init_project
```

This creates the project structure (`data/`, `outputs/`, `training_sets/`, `audioloop.yaml`).

### Step 3: Prepare Data

Download your dataset and place audio files in the project's `data/` directory, then generate spectrograms:

```bash
# With audioloop activated, from your project directory:
python -m audioloop.create_specs --dataset urbansound8k
```

### Step 4: Run Your First Workflow

```bash
python -m audioloop.utils.create_bootstrap_set --class-name siren --n 50
python -m audioloop.automated_workflow --class-name siren --cycles 2 --evaluation-mode --auto-label
```

### Step 5: Explore

See the detailed guides below for specific workflows and advanced usage.

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
- **TorchAudio + TorchCodec**: Audio processing and spectrogram generation
- **NumPy**: Numerical operations
- **TQDM**: Progress tracking
- **Ruff**: Code formatting and linting

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Citation

[Add citation information if applicable]
