import torch
import torch.nn as nn
import torch.nn.functional as F


class FxNet(nn.Module):
    """
    2D CNN architecture for guitar effect classification.
    Based on Comunità, Stowell, and Reiss (2021).
    """

    def __init__(self, n_classes: int = 12):
        super().__init__()
        self.n_classes = n_classes

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.batchNorm1 = nn.BatchNorm2d(num_features=6)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.batchNorm2 = nn.BatchNorm2d(num_features=12)

        # Fully Connected Layers
        # Input: 128 Mel bands x 87 frames -> Conv1 -> Pool -> Conv2 -> Pool -> 12 * 29 * 18 = 6264
        self.fc1 = nn.Linear(in_features=12 * 29 * 18, out_features=120)
        self.batchNorm3 = nn.BatchNorm1d(num_features=120)

        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.batchNorm4 = nn.BatchNorm1d(num_features=60)

        # Output Classification Layer
        self.out = nn.Linear(in_features=60, out_features=self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Conv Block 1
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv Block 2
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense Block 1
        x = self.fc1(x)
        x = self.batchNorm3(x)
        x = F.relu(x)

        # Dense Block 2
        x = self.fc2(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        # Logits
        logits = self.out(x)
        return logits
