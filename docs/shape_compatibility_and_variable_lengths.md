# Shape Compatibility and Variable Length Spectrograms

AudioLoop supports flexible spectrogram dimensions and automatic dataset/model compatibility checking. This enables domain-specific audio processing while preventing runtime failures from incompatible combinations.

## Overview

### Shape Compatibility System
- **Datasets declare output shapes** using `get_output_shape()`
- **Models declare input requirements** using `can_handle_shape()`
- **Training pipeline automatically validates** compatibility before starting
- **Clear error messages** guide users to compatible alternatives

### Variable Length Spectrograms
- **Natural audio durations preserved** (no forced padding)
- **Outlier handling** crops extremely long audio to reasonable limits
- **Implicit temporal augmentation** through natural length variation
- **Dynamic batch padding** handles different lengths within batches

## Shape Compatibility Interface

### Dataset Side: Declaring Output Shape

Datasets implement `get_output_shape()` to declare what tensor dimensions they produce:

```python
class MyDatasetConfig(DatasetConfig):
    def get_output_shape(self) -> tuple[int, ...]:
        return (64, -1)  # 64 frequency bins, variable time dimension
```

**Sentinel Values:**
- **`-1`** indicates a variable dimension (typically time)
- **Positive integers** indicate fixed dimensions

### Model Side: Declaring Input Requirements

Models implement `can_handle_shape()` to specify compatibility:

```python
class MyCNN(AudioLoopModel):
    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        return len(shape) == 2  # Can handle any 2D tensor
```

**Common Patterns:**

```python
# CNN with adaptive pooling - any 2D shape
def can_handle_shape(self, shape):
    return len(shape) == 2

# MLP requiring exact element count  
def can_handle_shape(self, shape):
    return math.prod(shape) == 127104  # 128 * 993

# RNN requiring fixed feature dimension, variable sequence
def can_handle_shape(self, shape):
    return len(shape) == 2 and shape[1] == 128
```

### Training Pipeline Integration

The training pipeline automatically checks compatibility:

```python
# Get dataset and model
dataset_shape = dataset_config.get_output_shape()
model = create_model(...)

# Automatic compatibility check
if not model.can_handle_shape(dataset_shape):
    raise ValueError(f"Model cannot handle shape {dataset_shape}")
```

**Error Messages:**
```
Model 'audio_mlp' cannot handle tensors with shape (128, -1) from dataset 'whale_calls'.
Available models: ['cnn5layer', 'simplecnn'].
Try a different model with --model-type or use a compatible dataset.
```

## Variable Length Spectrograms

### Current Behavior

AudioLoop now preserves natural audio durations instead of forcing all spectrograms to the same length:

#### Before (Fixed Length):
- All audio → exactly 993 or 2048 frames
- Short audio gets zero-padded (artificial silence)
- Long audio gets cropped (loses information)

#### After (Variable Length):
- Short audio → preserved at natural length (e.g., 400 frames)
- Medium audio → preserved at natural length (e.g., 1200 frames)  
- Outlier audio → cropped to reasonable maximum (e.g., 2048 frames)

### Dataset Configuration

Datasets control length handling via `fix_spectrogram_length()`:

```python
def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
    """Crop outliers but preserve natural variation."""
    current_length = spec.shape[-1]
    max_length = self.fixed_length  # Use as maximum, not target
    
    # Only crop outliers - preserve natural lengths
    if current_length > max_length:
        start_idx = (current_length - max_length) // 2
        spec = spec[..., start_idx : start_idx + max_length]
    
    return spec  # No padding for short spectrograms
```

### Batch Processing

The `variable_length_collate_fn` handles different lengths within batches:

```python
# Input batch with different lengths
batch = [
    (128, 400),   # Short spectrogram
    (128, 800),   # Medium spectrogram
    (128, 1200),  # Long spectrogram
]

# Output: padded to max length in batch
result = variable_length_collate_fn(batch)
# Shape: (3, 128, 1200) - all padded to longest (1200)
```

**Benefits:**
- **Within-batch consistency** for PyTorch tensor operations
- **Between-batch variation** for temporal augmentation
- **Minimal padding** (only to batch maximum, not global maximum)

## Benefits

### Domain-Specific Optimization
Different audio domains can use appropriate spectrogram dimensions:

```python
# Whale calls: Lower frequency range, longer time
class WhaleCallConfig(DatasetConfig):
    def get_output_shape(self):
        return (64, -1)  # 64 mels, variable time up to 1500 frames

# Bird songs: Higher frequency range, shorter time  
class BirdSongConfig(DatasetConfig):
    def get_output_shape(self):
        return (128, -1)  # 128 mels, variable time up to 800 frames
```

### Improved Model Training
- **Natural variation** improves model generalization
- **No artificial padding artifacts** from zero-padding short audio
- **No information loss** from cropping medium-length audio
- **Implicit temporal augmentation** through length diversity

### Automatic Validation
- **Prevents runtime crashes** from dimension mismatches
- **Clear error messages** with suggested fixes
- **Early failure** before expensive training starts
- **Future-proof** for non-CNN models (MLPs, Transformers, etc.)

## Examples

### Compatible Combinations

Current models work with all current datasets:

```bash
# All these combinations work
✅ fsd50k (128, -1) + cnn5layer
✅ fsd50k (128, -1) + simplecnn  
✅ urbansound8k (128, -1) + cnn5layer
✅ urbansound8k (128, -1) + simplecnn
```

### Incompatible Example

A hypothetical MLP model requiring exact input size:

```python
class AudioMLP(AudioLoopModel):
    def can_handle_shape(self, shape):
        return math.prod(shape) == 127104  # Exactly 128 * 993
        
# This would fail:
❌ fsd50k (128, -1) + audio_mlp
# Error: Cannot determine exact size from variable dimension
```

### Custom Dataset Example

```python
class MarineAcousticsConfig(DatasetConfig):
    """Optimized for underwater whale call detection."""
    
    fixed_length = 1500  # Max length for outlier cropping
    
    def get_output_shape(self):
        return (64, -1)  # Fewer mels (10-1000 Hz range), variable time
        
    def create_spectrogram_transform(self):
        return nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=2000,    # Lower sample rate for underwater
                n_mels=64,           # Focus on whale frequency range
                f_min=10.0,          # Blue whale calls start at ~10 Hz
                f_max=1000.0,        # Humpback upper range
            ),
            LogNormalize(top_db=80),
        )
```

## Migration from Fixed Lengths

### Existing Behavior Preserved
- **Maximum length limits** still prevent outliers (FSD50K: 2048, UrbanSound8K: 993)
- **Same model architectures** work without changes (adaptive pooling)
- **Same CLI commands** work without modification

### New Capabilities Added  
- **Natural length variation** enables better model training
- **Domain-specific optimization** through custom shape declarations
- **Automatic compatibility checking** prevents configuration errors
- **Future model support** for architectures requiring specific input dimensions

### No Breaking Changes
- **Existing datasets** return `(128, -1)` instead of `(128, 993)`
- **Existing models** accept any 2D shape via adaptive pooling
- **Existing workflows** continue working with improved training

## Technical Details

### Collate Function Selection
AudioLoop automatically uses `variable_length_collate_fn` for all training, which works for both fixed and variable length spectrograms.

### Memory Efficiency
- **Minimal padding**: Only pad within each batch, not to global maximum
- **Natural lengths**: No wasted memory on unnecessary padding
- **Batch grouping**: Future optimization could group similar lengths

### Model Architecture Support
- **CNNs with adaptive pooling**: Handle any input size ✅
- **Fully connected networks**: Require exact input size ⚠️  
- **RNNs**: Can handle variable sequence lengths ✅
- **Transformers**: May require sequence length limits ⚠️

The shape compatibility system ensures only compatible combinations are used, preventing runtime failures while enabling domain-specific optimizations.