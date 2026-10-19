import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, residual: bool = True):
        """residual=True: the network learns a correction added to its input y."""
        super().__init__()

        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if residual:
            nn.init.zeros_(self.conv5.weight)
            nn.init.zeros_(self.conv5.bias)
            # the last layer starts at zero, so the network is initially the identity

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(y))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return y + x if self.residual else x  # residual: x is a correction added to the input
