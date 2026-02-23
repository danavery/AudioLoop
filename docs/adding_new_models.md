# Adding New Models to AudioLoop

This guide explains how to add new models to the AudioLoop active learning framework. AudioLoop uses a pluggable model architecture that allows you to easily integrate custom PyTorch models while maintaining full compatibility with PyTorch conventions and the existing training/inference pipeline.

## Overview

AudioLoop uses a minimal abstract base class (`AudioLoopModel`) that extends `nn.Module` with just metadata and shape compatibility requirements. All models use standard PyTorch patterns while being automatically discoverable by the framework.

## The AudioLoopModel Interface

All models in AudioLoop must implement the `AudioLoopModel` abstract base class:

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

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

AudioLoop models are standard PyTorch modules with two additional methods: one for metadata and one for shape compatibility checking.

## Key Design Principles

- **Standard PyTorch Interface**: Models use the normal `forward(x: torch.Tensor)` signature
- **Shape Compatibility**: Models declare what input shapes they can handle
- **Ecosystem Compatibility**: Works with PyTorch hooks, torchscript, optimization tools
- **Automatic Discovery**: Just create a file - no manual registration needed
- **Metadata Preservation**: Constructor parameters are automatically saved/restored
- **Flexible Parameters**: Use `**kwargs` pattern for model-specific options

## Adding a Custom PyTorch Model

### Step 1: Create Your Model Class

Create a new file in `src/audioloop/models/` (e.g., `my_model.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_loop_model import AudioLoopModel


class MyCustomModel(AudioLoopModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        
        # Extract model-specific parameters
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.dropout_rate = kwargs.get('dropout_rate', 0.1)
        
        # Build your model architecture
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        # Add channel dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_model_info(self) -> dict:
        """Get model metadata - all constructor params are auto-saved."""
        return {
            "model_type": "my_custom_model",  # Must match filename
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
    
    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        """Declare what input shapes this model can handle."""
        # This CNN can handle any 2D spectrogram via adaptive pooling
        return len(shape) == 2
```

### Step 2: Implement Shape Compatibility

The `can_handle_shape()` method is crucial for preventing runtime errors. Different model architectures have different requirements:

```python
# CNN with adaptive pooling - any 2D shape
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return len(shape) == 2

# MLP requiring exact input size
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    import math
    return math.prod(shape) == self.required_input_size

# RNN requiring fixed feature dimension, variable sequence
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and shape[1] == 128  # (seq_len, features)

# Transformer with sequence length limits
def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    seq_len, features = shape
    return features == 128 and seq_len <= self.max_seq_length
```

**Shape Format:**
- Shapes are tuples excluding the batch dimension
- `-1` indicates variable dimensions (common for time axis)
- Models should handle sentinel values appropriately

### Step 3: Test Your Model

Create a simple test to verify your model works:

```python
# tests/test_my_model.py
import torch
from audioloop.models.my_model import MyCustomModel

def test_my_custom_model():
    model = MyCustomModel(num_classes=10)
    
    # Test shape compatibility
    assert model.can_handle_shape((128, 993)) is True
    assert model.can_handle_shape((128, -1)) is True
    assert model.can_handle_shape((100,)) is False
    
    # Test forward pass
    x = torch.randn(2, 128, 500)  # Variable length
    output = model(x)
    
    assert output.shape == (2, 10)
    assert not torch.isnan(output).any()
```

### Step 4: Use Your Model

Your model is now automatically discoverable and ready to use:

```bash
# List available models (your model will appear automatically)
python -m audioloop.train --list-models

# Train with your custom model
python -m audioloop.train training_set_v1.csv --model-type my_custom_model

# Pass custom parameters to your model
python -m audioloop.train training_set_v1.csv --model-type my_custom_model \\
  --model-kwargs '{"hidden_size": 512, "dropout_rate": 0.2}'
```

The system will use your `can_handle_shape()` method to validate compatibility with the dataset before training starts.