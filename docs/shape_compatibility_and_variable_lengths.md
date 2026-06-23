# Shape Compatibility and Variable-Length Features

AudioLoop validates feature/model compatibility at startup. The feature extractor declares the output shape it produces, models declare what shapes they accept, and the training pipeline checks compatibility before training begins.

Two extractor families exist today, and they produce tensors of different rank:

- **Spectrogram extractor** (default): variable-length 2D tensors `(n_mels, -1)`, where `-1` is the time axis.
- **Embedding extractor**: fixed-length 1D vectors `(D,)` — e.g. `(768,)` from a frozen wav2vec2 backbone.

The same compatibility machinery routes each to a model that accepts its rank: 2D spectrograms to the CNNs, 1D embeddings to the linear probe. Selecting between them is the `feature_extractor_type` config knob (see the [User Manual](user_manual.md#feature-extraction)).

## Feature Side: `get_output_shape()`

The feature extractor returns its tensor shape (excluding batch dimension). The sentinel value `-1` marks a variable dimension (typically time):

```python
# SpectrogramExtractor — variable time
def get_output_shape(self) -> tuple[int, ...]:
    return (self.n_mels, -1)  # e.g. (128, -1) — 128 mel bins, variable time

# EmbeddingExtractor — fixed-width pooled vector, no variable axis
def get_output_shape(self) -> tuple[int, ...]:
    return (768,)  # wav2vec2 hidden size; no -1, so no length handling applies
```

The embedding extractor pools its backbone's per-frame outputs to a single vector per file, so its shape has no `-1` and the length-handling below (crop / pad) is a no-op for it.

## Crop-Not-Pad Behavior: `_fix_length()`

Spectrograms shorter than the configured maximum are kept at their natural length (no zero-padding). Only outliers exceeding the max are center-cropped:

```python
def _fix_length(self, spec: torch.Tensor) -> torch.Tensor:
    current_length = spec.shape[-1]  # Time dimension is last
    if current_length > self.max_spectrogram_length:
        start_idx = (current_length - self.max_spectrogram_length) // 2
        spec = spec[..., start_idx : start_idx + self.max_spectrogram_length]
    return spec  # Short spectrograms returned as-is
```

## Model Side: `can_handle_shape()`

Models declare what input shapes they support. The CNN models use adaptive pooling, so they accept any 2D tensor:

```python
# CNN — any 2D shape works via adaptive pooling
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return len(shape) == 2
```

`LinearProbe` is the mirror case: it consumes a 1D embedding and rejects 2D spectrograms, simply by checking rank:

```python
# LinearProbe — 1D embeddings only
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return len(shape) == 1
```

This rank check is what makes the two paths self-routing: a spectrogram dataset (`(128, -1)`) fails fast against `LinearProbe`, and an embedding dataset (`(768,)`) fails fast against the CNNs, before any training starts. A model needing an exact element count (rather than just a rank) would compare `math.prod(shape)` and reject shapes containing `-1`.

## Batch Processing: `variable_length_collate_fn`

Within each batch, spectrograms are padded to the length of the longest item in that batch. This gives within-batch tensor consistency while preserving between-batch length variation:

```python
# Three spectrograms with different time dimensions:
#   (128, 400), (128, 800), (128, 1200)
# After collation → batch tensor shape: (3, 128, 1200)
```

Fixed-length 1D embeddings are already equal-length, so the same collate path stacks them with no padding (three `(768,)` vectors → `(3, 768)`).

See `src/audioloop/utils/data_utils.py` for the implementation.

## Compatibility Error

If a model can't handle a dataset's shape, training fails immediately with a message like:

```
Model 'audio_mlp' cannot handle tensors with shape (128, -1) from dataset 'fsd50k'.
Available models: ['cnn5layer', 'cnn7layer', 'simplecnn'].
Try a different model with --model-type or use a compatible dataset.
```
