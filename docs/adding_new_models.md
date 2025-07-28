# Adding New Models to AudioLoop

This guide explains how to add new models to the AudioLoop active learning framework. AudioLoop uses a pluggable model architecture that allows you to easily integrate custom PyTorch models while maintaining full compatibility with PyTorch conventions and the existing training/inference pipeline.

## Overview

AudioLoop uses a minimal abstract base class (`AudioLoopModel`) that extends `nn.Module` with just metadata requirements. All models use standard PyTorch patterns while being automatically discoverable by the framework.

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
```

That's it! AudioLoop models are standard PyTorch modules with just one additional method for metadata.

## Key Design Principles

- **Standard PyTorch Interface**: Models use the normal `forward(x: torch.Tensor)` signature
- **Ecosystem Compatibility**: Works with PyTorch hooks, torchscript, optimization tools
- **Automatic Discovery**: Just create a file - no manual registration needed
- **Metadata Preservation**: Constructor parameters are automatically saved/restored
- **Flexible Parameters**: Use `**kwargs` pattern for model-specific options

## Adding a Custom PyTorch Model

### Step 1: Create Your Model Class

Create a new file in `audioloop/audioloop/models/` (e.g., `my_model.py`):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_loop_model import AudioLoopModel


class MyCustomModel(AudioLoopModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        
        # Extract model-specific parameters from kwargs
        self.dropout_rate = kwargs.get('dropout_rate', 0.5)
        
        # Define your architecture here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, height, width) - channel dim will be added
               
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Add channel dimension if needed (for spectrograms)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        # Standard CNN forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.flatten(1)  # Better than view for dynamic shapes
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "my_model",  # Should match filename
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
```

### Step 2: Test Your Model

Create a simple test to verify your model works:

```python
# Test your model with standard PyTorch interface
model = MyCustomModel(num_classes=2, dropout_rate=0.3)

# Test forward pass with tensor input (standard PyTorch)
x = torch.randn(4, 128, 100)  # 4 spectrograms (batch, height, width)
outputs = model(x)  # Standard PyTorch calling convention
print(f"Output shape: {outputs.shape}")  # Should be (4, 2)

# Test that model info contains constructor parameters
info = model.get_model_info()
print(f"Model info: {info}")
assert info['dropout_rate'] == 0.3
```

### Step 3: Register Your Model

AudioLoop uses a dynamic model registry that automatically discovers models in the `audioloop/models/` directory. No registration is required - just place your model file in the correct location.

**File naming convention:**
- `my_model.py` → model name `"my_model"`
- Class name can be anything (e.g., `MyCustomModel`, `ResNet50`, etc.)
- The registry automatically finds any class that inherits from `AudioLoopModel`

### Step 4: Use Your Model

Your model is now automatically available throughout AudioLoop:

```bash
# List all available models
python -m audioloop.train --list-models

# Train with your custom model (basic usage)
python -m audioloop.train training_set.csv --model-type my_model

# Use in active learning (will automatically use the trained model)
python -m audioloop.active_learning --class-name Drill --run-number 1
```

You can also use it programmatically with custom parameters:

```python
from audioloop.config import AudioLoopConfig
from audioloop.training_core import run_training

# Configure training with your model and custom parameters
config = AudioLoopConfig(
    model_type="my_model",
    model_kwargs={
        "dropout_rate": 0.3,  # Custom parameter
        "other_param": "value"
    }
)

# Train - parameters are automatically saved and restored
run_training(config, labels_file="training_set.csv", version=1)
```

## Adding a HuggingFace Model

### Step 1: Install Dependencies

```bash
pip install transformers
```

### Step 2: Create Your HuggingFace Model Wrapper

For example, integrating the Audio Spectrogram Transformer:

```python
import torch
import torch.nn as nn
from transformers import ASTModel, ASTFeatureExtractor

from .audio_loop_model import AudioLoopModel


class AudioSpectrogramTransformer(AudioLoopModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = kwargs.get('model_name', "MIT/ast-finetuned-audioset-10-10-0.4593")
        
        # Load pre-trained HuggingFace model
        self.backbone = ASTModel.from_pretrained(self.model_name)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(self.model_name)
        
        # Add classification head for your task
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass.
        
        Args:
            x: Input tensor (AudioLoop will handle data extraction from batches)
               
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Convert AudioLoop spectrograms to HuggingFace format
        # Note: This example shows the concept - actual preprocessing 
        # would depend on your specific HuggingFace model requirements
        
        # Process with HuggingFace feature extractor if needed
        # For this example, assume x is already in the right format
        
        # Forward through HuggingFace model
        outputs = self.backbone(x)
        
        # Extract features (model-specific)
        features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        
        # Classification
        return self.classifier(features)
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "audio_spectrogram_transformer",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
```

**Note**: The above example shows the structure, but HuggingFace models often require specific preprocessing. You may need to customize the data pipeline in AudioLoop to work with your specific HuggingFace model's expected input format.

## Data Pipeline Integration

### Current System: Spectrograms
AudioLoop currently processes audio into spectrograms using `create_all_specs.py`. The training pipeline extracts spectrograms from batches and passes them as tensors to your model:

```python
# In training_core.py, your model receives:
features = batch["data"].to(device)  # Tensor of spectrograms
outputs = model(features)  # Standard PyTorch call
```

### Spectrogram Input Format
Your model will receive spectrograms as tensors with shape:
- `(batch_size, height, width)` - raw spectrograms
- Models typically add channel dimension: `x.unsqueeze(1)` → `(batch_size, 1, height, width)`

### Custom Preprocessing
If your model needs different preprocessing, you can handle it in the `forward()` method:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Custom preprocessing here
    x = your_preprocessing_function(x)
    
    # Your model's forward pass
    return your_model_forward(x)
```

## Best Practices

### 1. Input Validation
Validate inputs in your forward method:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Validate input shape
    if x.ndim not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D")
    
    # Add channel dimension if needed
    if x.ndim == 3:
        x = x.unsqueeze(1)
    
    # Rest of forward pass...
```

### 2. Use **kwargs Pattern
Make your models flexible with the **kwargs pattern:

```python
def __init__(self, num_classes: int, **kwargs):
    super().__init__()
    self.num_classes = num_classes
    
    # Extract optional parameters with defaults
    self.dropout_rate = kwargs.get('dropout_rate', 0.5)
    self.hidden_size = kwargs.get('hidden_size', 128)
    self.activation = kwargs.get('activation', 'relu')
```

### 3. Complete Metadata
Include all constructor parameters in `get_model_info()`:

```python
def get_model_info(self) -> dict:
    return {
        "model_type": "my_model",  # Should match filename
        "num_classes": self.num_classes,
        "dropout_rate": self.dropout_rate,  # All constructor params
        "hidden_size": self.hidden_size,
        "activation": self.activation,
        "num_parameters": sum(p.numel() for p in self.parameters()),
        "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
    }
```

### 4. Efficient Activations
Use functional activations for better performance:

```python
import torch.nn.functional as F

# Preferred: Functional (stateless, memory efficient)
x = F.relu(x)
x = F.dropout(x, p=0.5, training=self.training)

# Avoid: Module-based for simple activations
# self.relu = nn.ReLU()  # Creates unnecessary parameters
```

## Testing Your Model

### Unit Tests
Create tests for your model:

```python
import pytest
import torch
from audioloop.models.my_model import MyCustomModel

class TestMyCustomModel:
    def setup_method(self):
        self.model = MyCustomModel(num_classes=2, dropout_rate=0.3)
        
    def test_forward_pass(self):
        # Test with standard PyTorch tensor input
        x = torch.randn(4, 128, 100)  # Batch of spectrograms
        outputs = self.model(x)  # Standard PyTorch calling
        assert outputs.shape == (4, 2)
        assert not torch.isnan(outputs).any()
    
    def test_model_info_contains_constructor_params(self):
        info = self.model.get_model_info()
        assert info['num_classes'] == 2
        assert info['dropout_rate'] == 0.3
        assert 'num_parameters' in info
        
    def test_channel_dimension_handling(self):
        # Test that model handles both 3D and 4D inputs
        x_3d = torch.randn(2, 128, 100)  # No channel dim
        x_4d = torch.randn(2, 1, 128, 100)  # With channel dim
        
        out_3d = self.model(x_3d)
        out_4d = self.model(x_4d)
        
        assert out_3d.shape == (2, 2)
        assert out_4d.shape == (2, 2)
```

### Integration Tests
Test automatic save/load with AudioLoop:

```python
# Test automatic save/load (AudioLoop handles this)
from audioloop.config import AudioLoopConfig
from audioloop.training_core import run_training

config = AudioLoopConfig(
    model_type="my_model",
    model_kwargs={"dropout_rate": 0.2}
)

# AudioLoop automatically saves all constructor parameters
# and can reload the exact same model configuration
```

## Advanced Patterns

### Model Variants with Parameters
Create model variants using the **kwargs pattern:

```python
class FlexibleCNN(AudioLoopModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        
        # Size variants via parameters
        self.model_size = kwargs.get('model_size', 'medium')
        
        if self.model_size == 'small':
            channels = [16, 32]
        elif self.model_size == 'large':
            channels = [64, 128, 256]
        else:  # medium
            channels = [32, 64]
            
        # Build architecture based on size
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)
        
    def get_model_info(self) -> dict:
        return {
            "model_type": "flexible_cnn",
            "num_classes": self.num_classes,
            "model_size": self.model_size,  # Save parameter for reconstruction
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
```

Usage:
```bash
# Small variant
python -m audioloop.train training.csv --config '{"model_kwargs": {"model_size": "small"}}'

# Large variant  
python -m audioloop.train training.csv --config '{"model_kwargs": {"model_size": "large"}}'
```

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: AudioLoop passes spectrograms as `(batch, height, width)` - add channel dim if needed
2. **Missing Constructor Parameters**: If loading fails, ensure all parameters are in `get_model_info()`
3. **Import Errors**: Check that your model file is in `audioloop/models/` and follows naming conventions
4. **Parameter Reconstruction**: AudioLoop saves/loads all keys from `get_model_info()` except `num_parameters`

### Debugging Tips

1. **Test Model Creation**: Verify your model can be created with saved parameters
   ```python
   model = MyModel(num_classes=2, custom_param=42)
   info = model.get_model_info()
   # Remove num_parameters (not a constructor arg)
   constructor_args = {k: v for k, v in info.items() if k != 'num_parameters'}
   recreated = MyModel(**constructor_args)  # Should work
   ```

2. **Check Model Registry**: List available models to verify discovery
   ```bash
   python -m audioloop.train --list-models
   ```

3. **Test Forward Pass**: Ensure your model handles spectrogram inputs
   ```python
   model = MyModel(num_classes=2)
   x = torch.randn(1, 128, 100)  # Single spectrogram
   output = model(x)
   print(f"Output shape: {output.shape}")  # Should be (1, 2)
   ```

## Examples

See the existing models in `audioloop/models/` for reference:
- `cnn5layer.py`: Complex 5-layer CNN with adaptive BatchNorm (class: `CNN5Layer`)
- `simplecnn.py`: Lightweight 2-layer CNN example (class: `SimpleCnn`)

Both models demonstrate:
- Standard PyTorch `forward(x: torch.Tensor)` interface
- **kwargs parameter pattern for flexibility  
- Complete metadata in `get_model_info()`
- Proper channel dimension handling for spectrograms

## Summary

AudioLoop's pluggable model architecture is designed around standard PyTorch conventions:

**Key Requirements:**
- Inherit from `AudioLoopModel` (minimal abstract base class)
- Use standard `forward(x: torch.Tensor)` signature
- Include all constructor parameters in `get_model_info()`
- Use **kwargs pattern for flexibility

**Automatic Features:**
- Model discovery (just create the file)
- Parameter preservation (constructor args are saved/restored)
- CLI integration (--model-type my_model)
- Full PyTorch ecosystem compatibility

Whether you're adding a simple CNN or a complex transformer, the interface is standard PyTorch with minimal AudioLoop-specific requirements. This makes it easy to experiment with different architectures while maintaining the benefits of the active learning framework.