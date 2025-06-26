import torch.nn as nn


class SimpleCNN(nn.Module):
    """Lightweight CNN for audio classification.

    A simple 2-layer CNN with global average pooling, designed for
    binary classification of audio spectrograms. Much lighter than
    the 5-layer SoundCNN while maintaining good performance.
    """

    def __init__(self, num_classes=2):
        super().__init__()
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
