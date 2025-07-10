import torch
import torch.nn as nn
from .base import AudioLoopModel


class SimpleCNN(nn.Module, AudioLoopModel):
    """Lightweight CNN for audio classification.

    A simple 2-layer CNN with global average pooling, designed for
    binary classification of audio spectrograms. Much lighter than
    the 5-layer SoundCNN while maintaining good performance.
    """

    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               For spectrograms: (batch_size, 1, n_mels, time_frames)

        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def save_model(self, path: str) -> None:
        """Save model with metadata."""
        save_dict = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "model_type": "SimpleCNN"
        }
        torch.save(save_dict, path)

    @classmethod
    def load_model(cls, path: str, device: torch.device) -> "SimpleCNN":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)

        model = cls(num_classes=checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "SimpleCNN",
            "num_classes": self.num_classes,
            "num_parameters": sum(p.numel() for p in self.parameters())
        }
