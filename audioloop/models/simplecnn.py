import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_loop_model import AudioLoopModel


class SimpleCnn(AudioLoopModel):
    """Lightweight CNN for audio classification.

    A simple 2-layer CNN with global average pooling, designed for
    binary classification of audio spectrograms. Much lighter than
    the 5-layer CNN5Layer while maintaining good performance.
    """

    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, height, width) - channel dim will be added

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Add channel dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "simplecnn",
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        """Check if this model can handle tensors with the given shape."""
        # Can handle any 2D tensor via adaptive pooling
        # -1 in shape indicates variable dimension, which is fine for CNNs
        return len(shape) == 2
