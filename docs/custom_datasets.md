# Adding Custom Datasets to AudioLoop

How to use your own audio data with AudioLoop. This is the typical next step after running through the [quick start demo](../README.md).

## What You Need

- A directory of audio files (WAV, FLAC, MP3, etc.)
- A CSV file mapping filenames to class labels

Your CSV should look like:
```csv
filename,label
audio1.wav,speech
audio2.wav,music
audio3.wav,noise
```

## Quick Start

AudioLoop discovers datasets automatically by file naming convention. No registration required.

```bash
# 1. Create a datasets/ directory in your project root (alongside audioloop.yaml)
mkdir -p datasets

# 2. Copy the template
cp src/audioloop/datasets/templates/simple_audio_template.py datasets/my_dataset_config.py

# 3. Edit the copied file:
#    - Rename class from TemplateAudioConfig to MyDatasetConfig
#    - Update paths: data/YOUR_DATASET_NAME/ → data/my_dataset/
#    - Customize class vocabulary for your classes

# 4. Use immediately
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-classes
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-splits
```

## How Discovery Works

AudioLoop scans two locations for dataset config files, in order of precedence:

1. **Project-level** (recommended): `datasets/*_config.py` in your project root
2. **Built-in**: `src/audioloop/datasets/*_config.py` in the AudioLoop package

Naming conventions:
- **File naming**: `{dataset_name}_config.py` → dataset name `{dataset_name}`
- **Class naming**: `{DatasetName}Config` (e.g., `MyAudioConfig`, `CommonVoiceConfig`)

Project-level configs take precedence over built-in ones with the same name, so you can override built-in dataset behavior for your project.

## Verifying Your Dataset

```bash
# List all available datasets (yours should appear)
python -m audioloop.utils.create_bootstrap_set --help

# List classes in your dataset
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-classes

# List available splits
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-splits

# Create an initial training set
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --class-name speech --n 40

# Generate spectrograms
python -m audioloop.create_specs --dataset my_dataset
```

## Working with Large Datasets

For large datasets (100K+ files), you may want to create a subset first rather than working with the entire corpus:

```bash
# Create a manageable subset
python -m audioloop.create_subset --dataset my_dataset --class-name "Dog" --max-samples 1000

# Train directly on the subset — spectrograms are generated on demand
python -m audioloop.train subsets/my_dataset_dog_1000.csv
```

Subsets include an `audio_path` column that enables lazy spectrogram generation — no need to pre-generate specs for the entire dataset.

### Remote Deployment

For training on cloud pods or HPC clusters, create a self-contained spectrogram directory for efficient syncing:

```bash
# Create subset-specific specs directory (uses hard links — zero extra storage)
python -m audioloop.prepare_subset_specs subsets/my_dataset_dog_100000.csv

# Sync to remote
rsync -avz --no-o --no-g data/subset_specs/my_dataset_dog_100000/ user@pod:/workspace/data/specs/
```

## The DatasetConfig Interface

If you need more control than the template provides, here's what the `DatasetConfig` base class requires:

```python
class MyDatasetConfig(DatasetConfig):
    # Required: Define your dataset's properties
    def get_metadata_path(self) -> Path: ...
    def get_audio_dir(self) -> Path: ...
    def get_class_vocabulary(self) -> list[str]: ...
    def get_available_splits(self) -> list[str]: ...
    def get_default_split(self) -> str: ...

    # Required: Map your CSV format to AudioLoop's internal format
    def get_filename_column(self) -> str: ...
    def get_label_column(self) -> str: ...
```

See the existing dataset configs (`fsd50k_config.py`, `urbansound8k_config.py`, `audioset_config.py`) for complete examples of varying complexity.

## Programmatic Usage

```python
from audioloop.config import AudioLoopConfig

config = AudioLoopConfig(dataset="my_dataset")
dataset_config = config.get_dataset_config()

# Create subset programmatically
subset_path = dataset_config.create_subset(
    output_path=Path("subsets/my_subset.csv"),
    class_name="speech",
    max_samples=1000,
    positive_ratio=0.5,
    seed=42
)
```

## See Also
- [Dataset templates README](../src/audioloop/datasets/templates/README.md) — Template file documentation
- [CLI Reference](cli_reference.md) — Full `create_subset` and `create_bootstrap_set` parameters
