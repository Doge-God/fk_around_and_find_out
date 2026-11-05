"""Network architectures used by the experiments in this folder."""
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=10):
        super().__init__()
        sizes = [input_size, *hidden_sizes, num_classes]
        hidden_layers = [
            nn.Sequential(nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU())
            for a, b in zip(sizes[:-2], sizes[1:-1])
        ]
        self.layers = nn.ModuleList([*hidden_layers, nn.Linear(sizes[-2], sizes[-1])])

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.residual(x) + self.shortcut(x))


class ResNet18(nn.Module):
    """CIFAR-style ResNet-18: 3x3 first convolution and no max pool, so 32x32 inputs aren't shrunk early.

    self.layers holds the first convolution, the 8 residual blocks and the classifier, in order.
    """

    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        stem = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        blocks = [
            ResidualBlock(64, 64), ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2), ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2), ResidualBlock(512, 512),
        ]
        classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.layers = nn.ModuleList([stem, *blocks, classifier])

    def features(self, x):
        """Penultimate representation: the pooled output of the last block, just before the linear classifier."""
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1][:-1](x)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
