import torch
import torch.nn as nn

"""
3D ResNet Architecture for Medical Image Classification

Implements ResNet-18 and ResNet-34 variants adapted for 3D volumetric data.
Uses residual connections (skip connections) to enable training of deep networks.

Architecture:
    Input (1×128×128×128) 
    → Initial Conv (7×7×7) 
    → 4 Residual Layers (progressively downsampling)
    → Global Average Pooling 
    → Fully Connected Layer 
    → Output (num_classes)

Key Features:
    - Residual blocks prevent vanishing gradients
    - Configurable dropout for regularization
    - Kaiming weight initialization for ReLU networks
    - Supports binary and multi-class classification
"""

class BasicBlock3D(nn.Module):
    """
                 3D Residual Block - fundamental building block of ResNet.

        Implements: Conv3D → BN → ReLU → Dropout → Conv3D → BN → Add(skip) → ReLU

        The skip connection (residual) allows the block to learn incremental changes
        rather than full representations, enabling training of very deep networks.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for first convolution (default: 1)
            dropout (float): Dropout probability after first activation (default: 0.0)

        Attributes:
            conv1, conv2: 3×3×3 3D convolutions
            bn1, bn2: Batch normalization layers
            relu: ReLU activation
            dropout: Optional 3D dropout layer
            downsample: 1×1×1 conv for dimension matching (if needed)
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    """
    3D ResNet for volumetric medical image classification.

    Stacks multiple BasicBlock3D layers to create deep convolutional network.
    Architecture progressively downsamples spatial dimensions while increasing
    channel depth, then applies global pooling and classification head.

    Network flow:
        128³ → conv1(7×7×7, stride=2) → 64×64³
        → maxpool(3×3×3, stride=2) → 64×32³
        → layer1 (n blocks) → 64×32³
        → layer2 (n blocks, stride=2) → 128×16³
        → layer3 (n blocks, stride=2) → 256×8³
        → layer4 (n blocks, stride=2) → 512×4³
        → avgpool → 512×1³
        → dropout(0.5) → fc → num_classes

    Args:
        block_config (list): Number of blocks in each layer
                           [2,2,2,2] for ResNet-18, [3,4,6,3] for ResNet-34
        num_classes (int): Number of output classes (default: 2)
        dropout (float): Dropout probability in residual blocks (default: 0.3)

    Attributes:
        conv1: Initial 7×7×7 convolution
        layer1-4: Residual layer stacks
        avgpool: Global average pooling
        fc: Fully connected classification head
    """

    def __init__(self, block_config= None, num_classes=2, dropout=0.3):
        super().__init__()

        if block_config is None:
            block_config = [2, 2, 2, 2]


        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        #noinspection SpellCheckingInspection
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, block_config[0], stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, block_config[1], stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, block_config[2], stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, block_config[3], stride=2, dropout=dropout)
        # noinspection SpellCheckingInspection
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout_final = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()
    #noinspection PyMethodMayBeStatic
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [BasicBlock3D(in_channels, out_channels, stride, dropout)]

        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, stride=1, dropout=dropout))

        return nn.Sequential(*layers)
    #noinspection DuplicatedCode
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout_final(x)
        x = self.fc(x)

        return x

def resnet3d_18(num_classes=2, block_config=None, dropout=0.3):
    if block_config is None:
        block_config = [2, 2, 2, 2]
    return ResNet3D(block_config=block_config, num_classes=num_classes, dropout=dropout)


"""
    Create ResNet3D-18 model (18-layer configuration).

    Block config: [2, 2, 2, 2] = 8 BasicBlocks × 2 conv each = 16 layers
    Plus initial conv + FC = 18 layers total

    Args:
        num_classes (int): Number of output classes (default: 2)
        block_config (list): Custom block configuration (default: [2,2,2,2])
        dropout (float): Dropout probability (default: 0.3)

    Returns:
        ResNet3D: Configured ResNet-18 model
    """

def resnet3d_34(num_classes=2, block_config=None, dropout=0.3):
    if block_config is None:
        block_config = [3, 4, 6, 3]
    return ResNet3D(block_config=block_config, num_classes=num_classes, dropout=dropout)


"""
    Create ResNet3D-34 model (34-layer configuration).

    Block config: [3, 4, 6, 3] = 16 BasicBlocks × 2 conv each = 32 layers
    Plus initial conv + FC = 34 layers total

    Commonly used configuration with good balance of depth and efficiency.
    Approximately 63 million parameters for 2-class classification.

    Args:
        num_classes (int): Number of output classes (default: 2)
        block_config (list): Custom block configuration (default: [3,4,6,3])
        dropout (float): Dropout probability (default: 0.3)

    Returns:
        ResNet3D: Configured ResNet-34 model

    Example:
        >>> model = resnet3d_34(num_classes=2, dropout=0.3)
        >>> print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    """

