import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_loop_model import AudioLoopModel


class CNN5Layer(AudioLoopModel):
    description = "5-layer CNN with adaptive pooling (default)"

    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes

        # Extract model-specific parameters with defaults
        self.kernel_size = kwargs.get("kernel_size", (3, 3))
        dataset_size = kwargs.get("dataset_size")
        self.batchnorm_threshold = kwargs.get("batchnorm_threshold", 100.0)

        # Auto-decide BatchNorm unless explicitly overridden
        use_batchnorm = kwargs.get("use_batchnorm")
        if use_batchnorm is None:
            use_batchnorm = dataset_size is None or dataset_size >= self.batchnorm_threshold
        self.use_batchnorm = use_batchnorm

        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=self.kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=self.kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=self.kernel_size, stride=1, padding=padding
        )
        if self.use_batchnorm:
            self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
        )
        if self.use_batchnorm:
            self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
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

        # x shape: (batch, 1, n_mels, time_frames)
        x = self.conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        if self.use_batchnorm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        if self.use_batchnorm:
            x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = self.conv5(x)
        if self.use_batchnorm:
            x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)

        # Flatten for fully connected layers
        x = self.global_pool(x)  # shape: (batch, 256, 1, 1)
        x = x.flatten(1)  # shape: (batch, 256)
        x = self.fc1(x)
        x = F.relu(x)
        if not self.use_batchnorm:
            x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_type": "cnn5layer",
            "num_classes": self.num_classes,
            "kernel_size": self.kernel_size,
            "use_batchnorm": self.use_batchnorm,
            "batchnorm_threshold": self.batchnorm_threshold,
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    def can_handle_shape(self, shape: tuple[int, ...]) -> bool:
        """Check if this model can handle tensors with the given shape."""
        # Can handle any 2D tensor via adaptive pooling
        # -1 in shape indicates variable dimension, which is fine for CNNs
        return len(shape) == 2
