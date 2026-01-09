import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):

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

    def __init__(self, block_config= None, num_classes=2, dropout=0.3):
        super().__init__()

        if block_config is None:
            block_config = [2, 2, 2, 2]


        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # noinspection SpellCheckingInspection
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
    # noinspection PyMethodMayBeStatic
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [BasicBlock3D(in_channels, out_channels, stride, dropout)]

        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, stride=1, dropout=dropout))

        return nn.Sequential(*layers)
# noinspection DuplicatedCode
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

def resnet3d_34(num_classes=2, block_config=None, dropout=0.3):
    if block_config is None:
        block_config = [3, 4, 6, 3]
    return ResNet3D(block_config=block_config, num_classes=num_classes, dropout=dropout)