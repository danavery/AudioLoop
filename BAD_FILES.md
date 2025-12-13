# Handling Corrupt Audio Files

## Problem
Some audio files in datasets are corrupt and cause `torchaudio.load()` to segfault when attempting lazy spectrogram generation. These crashes cannot be caught with try/except.

## Solution
We maintain a list of known bad files in each dataset config and filter them out during processing.

## Adding a Bad File

When you discover a new corrupt file:

1. **Note the filename** from the crash (use debug mode if needed - see below)

2. **Add it to the dataset config** in `audioloop/datasets/<dataset>_config.py`:

```python
def get_bad_files(self) -> set[str]:
    """Return set of filenames known to crash during processing."""
    return {
        "Z3YaJ9Vi4lY.flac",  # Corrupt FLAC - crashes torchaudio.load()
        "your_new_bad_file.flac",  # Add description of why it's bad
    }
```

3. **Restart your workflow** - the file will now be automatically excluded

## Debug Mode: Finding Bad Files

To identify which file is crashing:

```bash
# Enable debug logging to see each file before it's processed
export AUDIOLOOP_DEBUG_FILES=1

# Run your command with --verbose flag
python -m audioloop.automated_workflow --verbose ...
```

The `--verbose` flag enables INFO-level logging (shows detailed progress).
The `AUDIOLOOP_DEBUG_FILES=1` environment variable logs every file before loading it.

Together, these will show you the last file logged before a crash - that's your culprit.

## Current Known Bad Files

### AudioSet
- `Z3YaJ9Vi4lY.flac` - Corrupt FLAC file (8.1K, "length unknown"), crashes torchaudio

## How It Works

1. Each `DatasetConfig` has a `get_bad_files()` method that returns a set of filenames
2. `active_learning_core.py` loads this list and filters out bad files when creating the dataset
3. Bad files are excluded from all processing (training, inference, etc.)
4. Debug mode logs each filename before processing to help identify new bad files

## Verification

The fix prevents the crash by excluding the bad file before it reaches the DataLoader:

```python
# Check bad files are loaded
from audioloop.config import AudioLoopConfig
config = AudioLoopConfig(dataset='audioset')
bad_files = config.get_dataset_config().get_bad_files()
print(bad_files)  # Should show: {'Z3YaJ9Vi4lY.flac'}
```
