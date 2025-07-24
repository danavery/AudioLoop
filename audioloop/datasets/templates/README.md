# Dataset Configuration Templates

This directory contains template files for creating custom dataset configurations. **Do not modify these templates directly** - instead, copy them to the parent `datasets/` directory and customize.

## Available Templates

### `simple_audio_template.py`
Template for the most common pattern: audio files in a folder + labels in a CSV file.

**Usage:**
```bash
# 1. Copy the template
cp audioloop/datasets/templates/simple_audio_template.py audioloop/datasets/my_dataset_config.py

# 2. Edit the copied file:
#    - Rename class from TemplateAudioConfig to MyDatasetConfig  
#    - Update paths to point to your data
#    - Customize class vocabulary

# 3. Use immediately
python -m audioloop.utils.start_labeling --dataset my_dataset --list-classes
```

**CSV Format:**
```csv
filename,label
audio1.wav,speech
audio2.wav,music
audio3.wav,noise
```

## Creating Your Own Templates

If you create a reusable template pattern, add it to this directory following the naming convention `pattern_name_template.py`.