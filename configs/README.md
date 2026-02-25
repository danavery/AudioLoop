# AudioLoop Configuration

AudioLoop uses a layered configuration system with two types of YAML files:

1. **Project defaults** (`audioloop.yaml`) - Settings that apply to all commands in your project
2. **Experiment configs** (`configs/*.yaml`) - Settings for specific experiments passed via `--config`

## Project Setup

Before running AudioLoop commands, initialize a project directory:

```bash
# Initialize in current directory
python -m audioloop.init_project

# Or initialize in a new directory
python -m audioloop.init_project ~/projects/dog-classifier
```

This creates the standard directory structure and an `audioloop.yaml` template:

```
my-project/
├── audioloop.yaml        # Project defaults (auto-loaded by all CLI commands)
├── data/
│   └── all_specs/        # Spectrograms
├── outputs/              # Model outputs, predictions
├── training_sets/        # Training CSVs
├── subsets/              # Dataset subsets
└── configs/              # Experiment-specific configs
```

### The `audioloop.yaml` File

The generated `audioloop.yaml` contains commented examples of all available settings:

```yaml
# AudioLoop Project Configuration
# Uncomment and modify any setting to change the default for this project.

# Dataset Configuration
dataset: fsd50k                  # Options: fsd50k, urbansound8k, audioset

# Training Parameters
# max_epochs: 1000
# batch_size: 32
# learning_rate: 0.001

# Active Learning
# selection_mode: entropy        # Options: entropy, confidence, basic_transition
# total_candidates: 50
```

All CLI commands automatically load settings from `audioloop.yaml` when run from within the project directory. You only need to specify parameters that differ from your project defaults.

### Working Outside the Project Directory

If you need to run commands from a different directory, set the environment variable:

```bash
export AUDIOLOOP_PROJECT_ROOT=/path/to/my-project
python -m audioloop.train training_set_v1.csv
```

## Experiment Configuration Files

This `configs/` directory contains YAML configuration files for specific experiments. These are useful for:
- Version controlling experiment settings
- Reusing configurations across multiple runs
- Sharing experiment setups with collaborators
- Reducing command-line complexity for complex experiments

### Quick Start

```bash
# Use an experiment config file (overrides project defaults)
python -m audioloop.automated_workflow --config configs/examples/search_mode.yaml

# Override specific values from CLI
python -m audioloop.automated_workflow --config configs/examples/minimal.yaml --cycles 10 --batch-size 64

# Use only project defaults (no --config needed)
python -m audioloop.automated_workflow --class-name Drill --cycles 3
```

## Experiment Config Structure

Experiment config files (passed via `--config`) support two formats:

### Two-Section Format (Recommended)

Separates workflow execution parameters from experiment configuration:

```yaml
workflow:
  class_name: "target_class"
  num_cycles: 5
  auto_label: true
  evaluation_mode: true
  # ... other workflow parameters

config:
  experiment_name: "exp_name"
  dataset: "fsd50k"
  max_epochs: 1000
  learning_rate: 0.001
  # ... other AudioLoopConfig parameters
```

### Flat Format (Alternative)

All parameters at root level:

```yaml
class_name: "target_class"
num_cycles: 5
experiment_name: "exp_name"
dataset: "fsd50k"
max_epochs: 1000
# ... all parameters at root level
```

## Configuration Precedence

Values are merged with this precedence (highest to lowest):

1. **CLI arguments** - Explicit command-line flags always win
2. **Experiment config** - Values from `--config` file (if provided)
3. **Project defaults** - Values from `audioloop.yaml` in project root
4. **Environment variables** - `AUDIOLOOP_*` environment variables
5. **Defaults** - Built-in defaults from `AudioLoopConfig` dataclass

**Example:**
```bash
# audioloop.yaml has:  learning_rate: 0.0005
# --config file has:   learning_rate: 0.001
# CLI overrides:       --learning-rate 0.01
# Result:              learning_rate = 0.01 (CLI wins)
```

**Typical usage patterns:**

| Scenario | Configuration approach |
|----------|----------------------|
| Team-wide defaults | Set in `audioloop.yaml`, commit to repo |
| Specific experiment | Create `configs/my_experiment.yaml` |
| Quick one-off test | Use CLI flags only |
| Override for one run | CLI flags override any YAML setting |

## Available Parameters

### Workflow Section

These parameters control workflow execution (specific to `automated_workflow.py`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `class_name` | string | **required** | Target audio class for classification |
| `num_cycles` | int | 3 | Number of active learning cycles to run |
| `start_cycle` | int | 1 | Starting cycle number (for resuming workflows) |
| `auto_label` | bool | false | Automatically label using ground truth |
| `evaluation_mode` | bool | false | Enable evaluation metrics with ground truth |
| `audio_dir` | string | null | Custom audio directory (optional) |
| `verbose` | bool | false | Enable verbose logging |

### Config Section

These parameters configure the AudioLoop experiment (all `AudioLoopConfig` dataclass fields). See `audioloop/config.py` for complete documentation.

#### Core Settings
- `experiment_name`: Experiment identifier (creates `outputs/{experiment_name}/`)
- `dataset`: Dataset name (`fsd50k`, `urbansound8k`, `audioset`)
- `subset_csv`: Path to dataset subset CSV file

#### Training Parameters
- `max_epochs`: Maximum training epochs (default: 1000)
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 1e-5)
- `seed`: Random seed for reproducibility (default: 42)
- `model_type`: Model architecture (default: `cnn5layer`)
- `class_weighting`: Class weighting mode (default: 0.70; options: `null`, `"adaptive"`, or float 0.0-1.0)

#### Learning Rate Scheduler
- `use_lr_scheduler`: Enable ReduceLROnPlateau (default: true)
- `lr_scheduler_factor`: LR reduction factor (default: 0.5)
- `lr_scheduler_patience`: Epochs before reducing LR (default: 5)
- `lr_scheduler_min_lr`: Minimum learning rate (default: 1e-6)

#### Stopping Criteria (Within-Epoch)
- `stopping_criterion_type`: Stopping criterion (`"plateau"`, `"accuracy"`)
- `patience`: Epochs without improvement (default: 20)
- `min_delta`: Minimum improvement threshold (default: 0.01)
- `accuracy_floor`: Min accuracy before patience counts (default: null)

#### Active Learning
- `total_candidates`: Candidates per cycle (default: 50)
- `positive_percentage`: Target positive ratio for stratification (default: null)
- `min_confidence`: Confidence threshold (default: 0.8)
- `selection_mode`: Selection strategy (`"entropy"`, `"confidence"`, `"basic_transition"`, etc.)
- `candidate_pool_multiplier`: Pool size multiplier (default: 5)

#### Selection Strategy (for `basic_transition` mode)
- `basic_transition_f1_threshold`: F1 threshold for strategy switch
- `basic_transition_confidence_threshold`: Confidence threshold
- `basic_transition_variance_threshold`: Variance threshold
- `auto_thresholds`: Auto-calculate thresholds from dataset
- `estimated_positive_pct`: Estimated positive % for auto-thresholds

#### Cycle Stopping (Cross-Cycle)
- `cycle_stopping_strategy`: Strategy (`"none"`, `"label"`, `"search"`)
- `cycle_patience`: Cycles without improvement (default: 5)
- `cycle_min_delta`: Min improvement to reset patience (default: 0.02)
- `cycle_min_cycles`: Min cycles before stopping (default: 10)
- `cycle_window`: Rolling average window (default: 3)
- `cycle_std_threshold`: Max std dev for stability (default: 0.08)
- `precision_floor`: Min precision for search mode (default: `"auto"`)

#### Evaluation
- `with_ground_truth`: Enable ground truth evaluation (default: false)

## Example Configurations

### Minimal Configuration (`examples/minimal.yaml`)

Quick start for basic experimentation:
```yaml
workflow:
  class_name: "siren"
  num_cycles: 3
  evaluation_mode: false

config:
  experiment_name: "quick_test"
```

### Search Mode (`examples/search_mode.yaml`)

Optimized for **recall** - finding all positive examples:
- Uses entropy-based selection for exploration
- Search mode cycle stopping (optimize recall with precision floor)
- Good for discovering rare instances in large datasets

### Label Mode (`examples/label_mode.yaml`)

Optimized for **F1 score** - balanced precision and recall:
- Uses adaptive strategy selection (basic_transition)
- Label mode cycle stopping (optimize F1)
- Adaptive class weighting for imbalanced data
- Good for general classification tasks

## Creating Custom Configs

1. **Start with an example** - Copy one of the example configs
2. **Modify parameters** - Adjust for your use case
3. **Test incrementally** - Start with few cycles to validate
4. **Version control** - Commit configs alongside code

**Example workflow:**
```bash
# Copy and modify
cp configs/examples/search_mode.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your settings

# Test with few cycles
python -m audioloop.automated_workflow --config configs/my_experiment.yaml --cycles 2

# Run full experiment
python -m audioloop.automated_workflow --config configs/my_experiment.yaml
```

## Tips and Best Practices

### When to Use Config Files
- ✅ Complex experiments with many parameters
- ✅ Reproducible research experiments
- ✅ Sharing experiment setups with collaborators
- ✅ Running parameter sweeps (future experiment runner)
- ⚠️ Quick one-off tests (CLI might be faster)

### Organizing Configs
```
configs/
├── examples/          # Built-in examples
├── experiments/       # Your research experiments
│   ├── dog_detection_v1.yaml
│   ├── dog_detection_v2.yaml
│   └── brass_classifier.yaml
└── sweeps/            # Parameter sweep configs
    └── learning_rate_sweep.yaml
```

### Common Patterns

**Incremental refinement:**
```yaml
# v1.yaml - baseline
config:
  experiment_name: "dog_v1"
  learning_rate: 0.001

# v2.yaml - tune learning rate
config:
  experiment_name: "dog_v2"
  learning_rate: 0.0005

# v3.yaml - add class weighting
config:
  experiment_name: "dog_v3"
  learning_rate: 0.0005
  class_weighting: "adaptive"
```

**Dataset-specific configs:**
```yaml
# audioset_template.yaml
config:
  dataset: "audioset"
  subset_csv: "subsets/audioset_{class}_10k.csv"
  max_epochs: 500
  selection_mode: "entropy"
```

### Validating Configs

The system will validate your config and provide helpful error messages:

**Invalid field names:**
```
ValueError: Invalid config fields in configs/test.yaml: invalid_param
Valid fields: accuracy_floor, auto_thresholds, batch_size, ...
```

**Missing required fields:**
```
Error: --class-name is required (either via CLI or --config file)
```

**Type errors:**
```
ValueError: max_epochs must be positive
```

## Troubleshooting

### No AudioLoop project found
```bash
RuntimeError: No AudioLoop project found in current directory.
Run 'python -m audioloop.init_project' to create one,
or set AUDIOLOOP_PROJECT_ROOT environment variable.
```
→ Run `python -m audioloop.init_project` to create the project structure, or set `AUDIOLOOP_PROJECT_ROOT` if working from a different directory.

### Config file not found
```bash
FileNotFoundError: Config file not found: configs/missing.yaml
```
→ Check the file path is correct and file exists

### YAML syntax error
```bash
yaml.YAMLError: mapping values are not allowed here
```
→ Check YAML syntax (indentation, colons, quotes)

### Parameter not being applied
- Check precedence: Is a CLI arg or experiment config overriding your project default?
- Check spelling: Invalid field names are silently ignored in flat format
- Use two-section format for better validation

### Workflow param vs config param confusion
- Workflow params: `class_name`, `num_cycles`, `auto_label`, etc.
- Config params: Everything else (training, active learning, etc.)
- Put them in the correct section for two-section format

## Additional Resources

- **Project initialization**: `python -m audioloop.init_project --help`
- **AudioLoopConfig documentation**: See `audioloop/config.py` docstring
- **Example usage**: See `automated_workflow.py` epilog for CLI examples
- **Developer guide**: See [`DEV_GUIDE.md`](../DEV_GUIDE.md) for architecture details

## Batch Runner

AudioLoop includes a batch runner for running multiple configs sequentially:

```bash
# Run all configs in a directory
python -m audioloop.batch_runner configs/my_experiment/*.yaml

# With a shared initial training set
python -m audioloop.batch_runner --initial-training-set training_sets/bootstrap.csv configs/*.yaml
```

Outputs are organized under a timestamped batch directory: `outputs/batch_YYYYMMDD_HHMMSS/{experiment_name}/`

See `python -m audioloop.batch_runner --help` for full options.
