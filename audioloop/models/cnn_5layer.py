import torch.nn as nn


class SoundCNN(nn.Module):
    def __init__(self, num_classes, kernel_size):
        super(SoundCNN, self).__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        # x shape: (batch, 1, n_mels, time_frames)

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.pool1(nn.ReLU()(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(nn.ReLU()(x))

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(nn.ReLU()(x))

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool4(nn.ReLU()(x))

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.pool5(nn.ReLU()(x))

        # Flatten for fully connected layers
        x = self.global_pool(x)  # shape: (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)  # shape: (batch, 64)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        # x = self.drop1(x)
        x = self.fc2(x)
        return x
