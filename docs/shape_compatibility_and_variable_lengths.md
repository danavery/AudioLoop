# Shape Compatibility and Variable-Length Spectrograms

AudioLoop validates dataset/model compatibility at startup and supports variable-length spectrograms. Datasets declare output shapes, models declare what shapes they accept, and the training pipeline checks compatibility before training begins.

## Dataset Side: `get_output_shape()`

Each dataset config returns its tensor shape (excluding batch dimension). The sentinel value `-1` marks a variable dimension (typically time):

```python
def get_output_shape(self) -> tuple[int, ...]:
    return (self.n_mels, -1)  # e.g. (128, -1) — 128 mel bins, variable time
```

## Crop-Not-Pad Behavior: `fix_spectrogram_length()`

Spectrograms shorter than the configured maximum are kept at their natural length (no zero-padding). Only outliers exceeding the max are center-cropped:

```python
def fix_spectrogram_length(self, spec: torch.Tensor) -> torch.Tensor:
    current_length = spec.shape[-1]
    max_length = self._max_spectrogram_length

    if current_length > max_length:
        start_idx = (current_length - max_length) // 2
        spec = spec[..., start_idx : start_idx + max_length]

    return spec  # Short spectrograms returned as-is
```

## Model Side: `can_handle_shape()`

Models declare what input shapes they support. All current CNN models use adaptive pooling, so they accept any 2D tensor:

```python
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return len(shape) == 2  # Any 2D shape works via adaptive pooling
```

A model requiring a fixed input size (e.g. an MLP) would check exact dimensions and reject shapes containing `-1`.

## Batch Processing: `variable_length_collate_fn`

Within each batch, spectrograms are padded to the length of the longest item in that batch. This gives within-batch tensor consistency while preserving between-batch length variation:

```python
# Three spectrograms with different time dimensions:
#   (128, 400), (128, 800), (128, 1200)
# After collation → batch tensor shape: (3, 128, 1200)
```

See `src/audioloop/utils/data_utils.py` for the implementation.

## Compatibility Error

If a model can't handle a dataset's shape, training fails immediately with a message like:

```
Model 'audio_mlp' cannot handle tensors with shape (128, -1) from dataset 'fsd50k'.
Available models: ['cnn5layer', 'cnn7layer', 'simplecnn'].
Try a different model with --model-type or use a compatible dataset.
```
