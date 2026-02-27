# Adding New Models to AudioLoop

AudioLoop uses a minimal abstract base class (`AudioLoopModel`) that extends `nn.Module` with two additional methods: one for metadata and one for shape compatibility checking. Models are automatically discovered by filename — no registration needed.

```python
class AudioLoopModel(nn.Module, ABC):
    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model metadata for tracking and logging."""
        pass

    @abstractmethod
    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        """Check if this model can handle tensors with the given shape."""
        pass
```

## Creating a Model

Create a new file in your project's `models/` directory (recommended) or in `src/audioloop/models/` (in-package). The filename becomes the model name for `--model-type`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# For project-level models (models/ in your project root):
from audioloop.models.audio_loop_model import AudioLoopModel

# For in-package models (src/audioloop/models/):
# from .audio_loop_model import AudioLoopModel


class MyCustomModel(AudioLoopModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # Extract model-specific parameters (passed via model_kwargs in audioloop.yaml)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)

        # Build your model architecture
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def get_model_info(self) -> dict:
        """Metadata saved alongside the model weights."""
        return {
            "model_type": "my_model",
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        """This CNN can handle any 2D spectrogram via adaptive pooling."""
        return len(shape) == 2
```

**Notes on the interface:**

- `forward()` uses the standard PyTorch signature. Input spectrograms are `(batch, n_mels, time)`.
- `get_model_info()` returns metadata that gets saved alongside the model weights when training completes. Include anything needed to reconstruct the model — this is not automatic.
- `can_handle_shape()` is called before training starts to verify the model is compatible with the dataset. The shape tuple excludes the batch dimension. A `-1` indicates a variable dimension — all built-in datasets return `(n_mels, -1)` since spectrogram length varies. Models with adaptive pooling can just check `len(shape) == 2`; models that need a fixed input size would need to reject shapes containing `-1`.

## Using Your Model

```bash
# List available models (your model will appear automatically)
python -m audioloop.train --list-models

# Train with your custom model
python -m audioloop.train training_set_v1.csv --model-type my_model
```

Pass custom parameters to your model via `model_kwargs` in `audioloop.yaml`:

```yaml
model_type: my_model
model_kwargs:
  hidden_size: 512
  dropout_rate: 0.2
```

These are passed as `**kwargs` to your model's constructor.

## See Also
- [User Manual: Training](user_manual.md#training) — training options and model types
- [Extending AudioLoop](extending.md) — general extensibility guide
- [Shape Compatibility](shape_compatibility_and_variable_lengths.md) — how variable-length spectrograms are handled
