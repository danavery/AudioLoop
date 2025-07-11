# Adding New Models to AudioLoop

This guide explains how to add new models to the AudioLoop active learning framework. AudioLoop uses a pluggable model architecture that allows you to easily integrate custom PyTorch models or HuggingFace models while maintaining compatibility with the existing training and inference pipeline.

## Overview

AudioLoop uses an abstract base class (`AudioLoopModel`) that defines a consistent interface for all models. This allows the training, inference, and active learning code to work with any model that implements this interface.

## The AudioLoopModel Interface

All models in AudioLoop must implement the `AudioLoopModel` abstract base class:

```python
from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class AudioLoopModel(ABC):
    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    @abstractmethod
    def prepare_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input batch for the model's expected format."""
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to disk with metadata."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, device: torch.device) -> "AudioLoopModel":
        """Load a model from disk."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Get model metadata."""
        pass
```

## Method Responsibilities

### `forward(batch: Dict[str, Any]) -> torch.Tensor`
- **Purpose**: Perform the forward pass through the model
- **Input**: Dictionary containing model inputs (typically with "data" key)
- **Output**: Logits tensor of shape `(batch_size, num_classes)`
- **Note**: This replaces the standard PyTorch `forward(x)` method

### `prepare_input(batch: Dict[str, Any]) -> Dict[str, Any]`
- **Purpose**: Convert raw batch data into the format expected by your model
- **Input**: Raw batch from the dataloader
- **Output**: Dictionary with model-ready inputs
- **Use Cases**: Device placement, data type conversion, preprocessing

### `get_device() -> torch.device`
- **Purpose**: Return the device the model is currently on
- **Typical Implementation**: `return next(self.parameters()).device`

### `save_model(path: str) -> None`
- **Purpose**: Save the model state and metadata to disk
- **Requirements**: Must save enough information to reconstruct the model

### `load_model(cls, path: str, device: torch.device) -> "AudioLoopModel"`
- **Purpose**: Class method to load a saved model
- **Requirements**: Must recreate the model and load its state

### `get_model_info() -> dict`
- **Purpose**: Return metadata about the model
- **Typical Contents**: Model type, architecture details, parameter count

## Adding a Custom PyTorch Model

### Step 1: Create Your Model Class

Create a new file in `audioloop/audioloop/models/` (e.g., `my_model.py`):

```python
import torch
import torch.nn as nn
from typing import Dict, Any

from .base import AudioLoopModel


class MyCustomModel(nn.Module, AudioLoopModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        
        # Define your architecture here
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass expecting spectrogram data."""
        features = batch["data"]
        
        # Add channel dimension if needed
        if features.ndim == 3:
            features = features.unsqueeze(1)
        
        # Your forward implementation
        x = self.pool(torch.relu(self.conv1(features)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def prepare_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for custom model."""
        features = batch["data"].to(self.get_device())
        return {"data": features}
    
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def save_model(self, path: str) -> None:
        """Save model with metadata."""
        save_dict = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "model_type": "MyCustomModel",
            # Add any other parameters needed to reconstruct the model
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> "MyCustomModel":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            num_classes=checkpoint["num_classes"],
            # Pass any other constructor parameters from checkpoint
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "MyCustomModel",
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
```

### Step 2: Test Your Model

Create a simple test to verify your model works:

```python
# Test your model
model = MyCustomModel(num_classes=2)
batch = {"data": torch.randn(4, 128, 100)}  # 4 spectrograms

# Test the interface
model_inputs = model.prepare_input(batch)
outputs = model.forward(model_inputs)
print(f"Output shape: {outputs.shape}")  # Should be (4, 2)
```

### Step 3: Integration

Your model can now be used with AudioLoop's training and inference pipeline:

```python
# In training_core.py or wherever you create models
from audioloop.models.my_model import MyCustomModel

model = MyCustomModel(num_classes=2)
# The rest of the training pipeline works automatically
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
from typing import Dict, Any

from .base import AudioLoopModel


class AudioSpectrogramTransformer(nn.Module, AudioLoopModel):
    def __init__(self, num_classes: int, model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Load pre-trained HuggingFace model
        self.backbone = ASTModel.from_pretrained(model_name)
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
        
        # Add classification head for your task
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through HuggingFace model."""
        inputs = batch["input_values"]
        
        # Forward through HuggingFace model
        outputs = self.backbone(inputs)
        
        # Extract features (model-specific)
        features = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        
        # Classification
        return self.classifier(features)
    
    def prepare_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for HuggingFace model."""
        # Convert AudioLoop spectrograms to HuggingFace format
        spectrograms = batch["data"]
        
        # Process with HuggingFace feature extractor
        processed = self.feature_extractor(
            spectrograms.squeeze(1).numpy(),  # Model-specific preprocessing
            return_tensors="pt",
            sampling_rate=16000
        )
        
        return {"input_values": processed.input_values.to(self.get_device())}
    
    def get_device(self) -> torch.device:
        """Get the device this model is on."""
        return next(self.parameters()).device
    
    def save_model(self, path: str) -> None:
        """Save HuggingFace model wrapper."""
        save_dict = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "backbone_state_dict": self.backbone.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "model_type": "AudioSpectrogramTransformer",
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load_model(cls, path: str, device: torch.device) -> "AudioSpectrogramTransformer":
        """Load HuggingFace model wrapper."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            num_classes=checkpoint["num_classes"],
            model_name=checkpoint["model_name"]
        )
        
        model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        model.to(device)
        return model
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "AudioSpectrogramTransformer",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
```

## Data Pipeline Considerations

### Current System: Spectrograms
AudioLoop currently processes audio into spectrograms using `create_all_specs.py`. Models that work with spectrograms (like CNNs and AST) can use the existing pipeline:

```bash
# Pre-process audio to spectrograms
python -m audioloop.create_all_specs

# Train with any spectrogram-compatible model
python -m audioloop.train training_set.csv
```

### Future: Raw Audio Models
For models that require raw audio (like Wav2Vec2, Whisper), the data pipeline would need to be extended. This is planned for future development.

## Best Practices

### 1. Input Validation
Always validate that inputs are in the expected format:

```python
def prepare_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    features = batch["data"]
    
    # Validate input shape
    if features.ndim not in [3, 4]:
        raise ValueError(f"Expected 3D or 4D input, got {features.ndim}D")
    
    # Rest of preprocessing...
```

### 2. Device Management
Always ensure tensors are on the right device:

```python
def prepare_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    features = batch["data"].to(self.get_device())
    return {"data": features}
```

### 3. Error Handling
Include proper error handling in your model loading:

```python
@classmethod
def load_model(cls, path: str, device: torch.device) -> "MyModel":
    try:
        checkpoint = torch.load(path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")
    
    # Validate checkpoint contents
    required_keys = ["model_state_dict", "num_classes", "model_type"]
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Missing required key in checkpoint: {key}")
    
    # Rest of loading logic...
```

### 4. Comprehensive Metadata
Include useful metadata for debugging and model management:

```python
def get_model_info(self) -> dict:
    return {
        "model_type": "MyModel",
        "num_classes": self.num_classes,
        "num_parameters": sum(p.numel() for p in self.parameters()),
        "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in self.parameters()) / (1024 * 1024),
        "architecture_details": {
            # Add architecture-specific details
        }
    }
```

## Testing Your Model

### Unit Tests
Create tests for your model:

```python
import unittest
import torch
from audioloop.models.my_model import MyCustomModel

class TestMyCustomModel(unittest.TestCase):
    def setUp(self):
        self.model = MyCustomModel(num_classes=2)
        self.batch = {"data": torch.randn(4, 128, 100)}
    
    def test_forward_pass(self):
        model_inputs = self.model.prepare_input(self.batch)
        outputs = self.model.forward(model_inputs)
        self.assertEqual(outputs.shape, (4, 2))
    
    def test_save_load(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt') as f:
            self.model.save_model(f.name)
            loaded_model = MyCustomModel.load_model(f.name, torch.device('cpu'))
            self.assertEqual(loaded_model.num_classes, 2)
```

### Integration Tests
Test with the actual AudioLoop pipeline:

```python
# Test with real training data
from audioloop.training_core import run_training

model = MyCustomModel(num_classes=2)
# Run a short training to verify integration
```

## Common Patterns

### Multiple Model Variants
You can create multiple variants of the same model:

```python
class MyModelSmall(MyCustomModel):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        # Smaller architecture
        
class MyModelLarge(MyCustomModel):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        # Larger architecture
```

### Model Factories
For complex model creation:

```python
def create_my_model(variant: str, num_classes: int) -> MyCustomModel:
    if variant == "small":
        return MyModelSmall(num_classes)
    elif variant == "large":
        return MyModelLarge(num_classes)
    else:
        raise ValueError(f"Unknown variant: {variant}")
```

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Ensure your model expects the right input shape
2. **Device Issues**: Always move tensors to the model's device
3. **Save/Load Failures**: Include all necessary information in checkpoints
4. **Missing Dependencies**: Install all required packages for HuggingFace models

### Debugging Tips

1. **Print Tensor Shapes**: Add debug prints to understand data flow
2. **Test in Isolation**: Test your model separately before integration
3. **Check Device Placement**: Verify all tensors are on the same device
4. **Validate Outputs**: Ensure outputs have the expected shape and range

## Examples

See the existing models in `audioloop/audioloop/models/` for reference:
- `cnn_5layer.py`: Complex CNN with batch normalization
- `simple_cnn.py`: Lightweight CNN example

## Contributing

When adding new models to the AudioLoop codebase:

1. Follow the existing code style and patterns
2. Add comprehensive docstrings
3. Include unit tests
4. Update this documentation if needed
5. Consider adding examples or demos

The pluggable model architecture makes AudioLoop extensible while maintaining consistency across all model types. Whether you're adding a simple CNN or a complex transformer, the interface remains the same, making it easy to experiment with different architectures in your active learning workflow.