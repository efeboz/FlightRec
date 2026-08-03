"""A compact ResNet-18 adapted to 32 by 32 inputs."""

from torch import Tensor, nn


class BasicBlock(nn.Module):
    """Two-convolution residual block."""

    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut: nn.Module
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the residual block."""
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + self.shortcut(inputs))


class CifarResNet(nn.Module):
    """ResNet-18 with a 3x3 stem and no max pooling."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.layer1 = self._layer(64, 64, 2, 1)
        self.layer2 = self._layer(64, 128, 2, 2)
        self.layer3 = self._layer(128, 256, 2, 2)
        self.layer4 = self._layer(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _layer(in_channels: int, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, channels, stride)]
        layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class logits."""
        features = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(inputs)))))
        return self.fc(self.pool(features).flatten(1))


def resnet18_cifar(num_classes: int = 10) -> CifarResNet:
    """Construct the case-study model."""
    return CifarResNet(num_classes)
