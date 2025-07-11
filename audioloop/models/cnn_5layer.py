import torch
import torch.nn as nn
from typing import Dict, Any

from .base import AudioLoopModel


class SoundCNN(nn.Module, AudioLoopModel):
    def __init__(
        self,
        num_classes,
        kernel_size=(3, 3),
        dataset_size=None,
        batchnorm_threshold=100.0,
        _use_batchnorm=None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.batchnorm_threshold = batchnorm_threshold

        # Auto-decide BatchNorm based on dataset size, unless explicitly overridden
        if _use_batchnorm is not None:
            use_batchnorm = _use_batchnorm
        else:
            use_batchnorm = dataset_size is None or dataset_size >= batchnorm_threshold
        self.use_batchnorm = use_batchnorm
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        if not self.use_batchnorm:
            self.dropout = nn.Dropout(
                0.5
            )  # Add dropout for regularization when not using BatchNorm
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Forward pass expecting spectrogram data."""
        features = batch["data"]

        # Add channel dimension if needed
        if features.ndim == 3:
            features = features.unsqueeze(1)

        # x shape: (batch, 1, n_mels, time_frames)
        x = features

        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)

        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.pool3(x)

        x = self.conv4(x)
        if self.use_batchnorm:
            x = self.bn4(x)
        x = nn.ReLU()(x)
        x = self.pool4(x)

        x = self.conv5(x)
        if self.use_batchnorm:
            x = self.bn5(x)
        x = nn.ReLU()(x)
        x = self.pool5(x)

        # Flatten for fully connected layers
        x = self.global_pool(x)  # shape: (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # shape: (batch, 256)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        if not self.use_batchnorm:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

    def prepare_input(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for custom CNN model."""
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
            "kernel_size": self.kernel_size,
            "use_batchnorm": self.use_batchnorm,
            "batchnorm_threshold": self.batchnorm_threshold,
            "model_type": "SoundCNN",
        }
        torch.save(save_dict, path)

    @classmethod
    def load_model(cls, path: str, device: torch.device) -> "SoundCNN":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device)

        # Create model with the saved BatchNorm setting (bypassing auto-decision)
        model = cls(
            num_classes=checkpoint["num_classes"],
            kernel_size=checkpoint["kernel_size"],
            batchnorm_threshold=checkpoint.get("batchnorm_threshold", 100),
            _use_batchnorm=checkpoint["use_batchnorm"],  # Use the saved BatchNorm setting
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "SoundCNN",
            "num_classes": self.num_classes,
            "kernel_size": self.kernel_size,
            "use_batchnorm": self.use_batchnorm,
            "batchnorm_threshold": self.batchnorm_threshold,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }
