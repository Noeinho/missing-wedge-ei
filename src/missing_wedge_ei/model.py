import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, residual: bool = True):
        """Residual = true si on veut faire apprendre au réseau la partie manquante uniquement."""
        super().__init__()

        self.residual = residual
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if residual:
            nn.init.zeros_(self.conv5.weight)
            nn.init.zeros_(
                self.conv5.bias
            )  # met à 0 poids et biais de la dernière couche. le _ est la convention pytorch pour modifier
            # au lieu de renvoyer un objet différent.
            # Cela signifie qu'initialement, 0 sort de residual.

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(y))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return (
            y + x if self.residual else x
        )  # si residual, le x appris par le réseau n'est que la partie manquante. on l'ajoute à l'entrée.
