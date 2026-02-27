# Adding Custom Datasets to AudioLoop

How to use your own audio data with AudioLoop.

## What You Need

- A directory of audio files (WAV, FLAC, MP3, etc.)
- A CSV file mapping filenames to class labels

The template expects a simple CSV format:
```csv
filename,label
audio1.wav,speech
audio2.wav,music
audio3.wav,noise
```

If your dataset already has its own metadata files in a different format, you can override `_load_metadata_for_split()` and `parse_metadata_row()` in your config to parse them directly — see [The DatasetConfig Interface](#the-datasetconfig-interface) below.

## Setup

AudioLoop discovers datasets automatically by file naming convention. No registration required.

```bash
# 1. Create a datasets/ directory in your project root (alongside audioloop.yaml)
mkdir -p datasets

# 2. Copy the template
cp src/audioloop/datasets/templates/simple_audio_template.py datasets/my_dataset_config.py

# 3. Edit the copied file (see "What to Customize" below)

# 4. Verify it works
python -m audioloop.utils.create_bootstrap_set --dataset my_dataset --list-classes

# 5. Generate spectrograms (or let them generate lazily during training)
python -m audioloop.create_specs --dataset my_dataset
```

### Creating Your Bootstrap Training Set

To start the active learning loop, you need an initial training set CSV with a small number of hand-labeled positive and negative examples. The format is:

```csv
filename,label,audio_path
clip_001.pt,1,/path/to/audio/clip_001.wav
clip_002.pt,0,/path/to/audio/clip_002.wav
```

Where `label` is `1` for positive (matches your target class) and `0` for negative. The `audio_path` column is optional but enables lazy spectrogram generation.

`create_bootstrap_set --class-name X --n 40` can generate this automatically, but only in evaluation mode where ground truth labels already exist. For real-world use, you'll need to create this CSV yourself from a small set of clips you've listened to and labeled.

## What to Customize

The setup steps above copy the template into your project's `datasets/` directory as a new file (e.g., `datasets/my_dataset_config.py`). This file defines your own `DatasetConfig` subclass. Rename the class from `TemplateAudioConfig` to match your dataset (the naming convention is `{DatasetName}Config`), then customize the fields at the top:

```python
class MyDatasetConfig(DatasetConfig):   # Was TemplateAudioConfig

    # Point to your data
    _dataset_csv = Path("data/my_dataset/labels.csv")
    _audio_root = Path("data/my_dataset/clips")
    _audio_extension = ".wav"

    # CSV column names (if different from defaults)
    _filename_column = "filename"
    _label_column = "label"

    # Your classes
    _class_vocabulary: ClassVar[dict[int, str]] = {
        0: "speech",
        1: "music",
        2: "noise",
    }

    # Audio processing (adjust if needed)
    _sample_rate = 22050
    _n_fft = 1024
    _hop_length = 512
    _n_mels = 128
    _max_spectrogram_length = 993
```

Everything below the customization section implements the `DatasetConfig` interface and typically doesn't need changes.

## How Discovery Works

AudioLoop scans two locations for dataset config files, in order of precedence:

1. **Project-level** (recommended): `datasets/*_config.py` in your project root
2. **Built-in**: `audioloop/datasets/*_config.py` in the AudioLoop package

Naming conventions:
- **File naming**: `{dataset_name}_config.py` → dataset name `{dataset_name}`
- **Class naming**: `{DatasetName}Config` (e.g., `MyDatasetConfig`, `CommonVoiceConfig`)

Project-level configs take precedence over built-in ones with the same name, so you can override built-in dataset behavior for your project.

## The DatasetConfig Interface

The template implements every required method from `DatasetConfig`. For most custom datasets, you only need to change the class attributes in [What to Customize](#what-to-customize). But if your dataset has a non-standard structure, you can override these methods:

**Metadata** (how your CSV/metadata is parsed):
- `_load_metadata_for_split(split)` — load all rows for a split
- `parse_metadata_row(row)` — parse one CSV row into `{filename, labels, audio_path, ...}`

**File paths** (where things live):
- `get_audio_path(filename, split, fold)` — resolve audio file location
- `get_spectrogram_path(filename, specs_dir)` — resolve spectrogram file location

**Classification** (how labels are interpreted):
- `is_positive_class(class_name, positive_class)` — determine if a class matches the positive class
- `get_binary_label(item, positive_class_id, positive_class_name)` — get 0/1 label for an item

**Audio processing** (spectrogram generation):
- `create_spectrogram_transform()` — PyTorch transform pipeline
- `fix_spectrogram_length(spec)` — handle length outliers
- `process_single_file(file_info, output_dir)` — full processing pipeline for one file

For split/fold customization, see [Splits and Folds](#splits-and-folds) below. See the built-in configs (`fsd50k_config.py`, `urbansound8k_config.py`, `audioset_config.py`) for examples of varying complexity.

## Splits and Folds

By default, the template treats your dataset as a single unit with one split called `"all"`. This is fine for most custom datasets. AudioLoop manages its own training sets internally and doesn't need pre-defined train/test splits.

If your dataset does have splits (e.g., `train`/`test`) or folds (e.g., for cross-validation), override these methods in your config:

```python
def get_available_splits(self) -> list[str]:
    return ["train", "test"]

def get_default_split(self) -> str:
    return "train"
```

The `split` parameter is passed through to `_load_metadata_for_split()` and `get_audio_path()`, so your implementations can use it to load the right files. See the built-in configs for examples: FSD50K uses splits (`dev`/`eval`), UrbanSound8K uses a `fold` field in its metadata.

## Working with Large Datasets

For large datasets (100K+ files), you may want to create a subset rather than working with the entire corpus:

```bash
# Create a manageable subset
python -m audioloop.create_subset --dataset my_dataset --class-name "Dog" --max-samples 1000

# Train directly on the subset — spectrograms are generated on demand
python -m audioloop.train subsets/my_dataset_dog_1000.csv
```

Subsets include an `audio_path` column that enables lazy spectrogram generation — no need to pre-generate specs for the entire dataset.

For training on cloud pods or HPC clusters, create a self-contained spectrogram directory for efficient syncing:

```bash
# Create subset-specific specs directory (uses hard links — zero extra storage)
python -m audioloop.prepare_subset_specs subsets/my_dataset_dog_100000.csv

# Sync to remote
rsync -avz --no-o --no-g data/subset_specs/my_dataset_dog_100000/ user@pod:/workspace/data/specs/
```

## See Also
- [Dataset templates README](../src/audioloop/datasets/templates/README.md) — Template file documentation
- Run `python -m audioloop.create_subset --help` or `python -m audioloop.utils.create_bootstrap_set --help` for full parameters
