"""
MedicalNet wrapper for 3D ResNet-18 pretrained on 23 medical imaging datasets.

This file mirrors the interface of resnet3d_OASIS.py so it slots into train.py
as a drop-in replacement. The only difference is the model architecture and the
pretraining domain (medical 3D scans vs Kinetics-400 videos).

Architecture source: Tencent/MedicalNet (https://github.com/Tencent/MedicalNet)
Based on: 3D-ResNets-PyTorch by Kensho Hara
Paper: Chen et al. (2019) "Med3D: Transfer Learning for 3D Medical Image Analysis"
       https://arxiv.org/abs/1904.00625

Usage:
    from medicalnet3d import MedicalNetConfig, build_medicalnet
    cfg = MedicalNetConfig(model_depth=18, pretrained_path="pretrained/resnet_18_23dataset.pth")
    model = build_medicalnet(cfg)
"""

from dataclasses import dataclass
from functools import partial
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuration dataclass (mirrors ModelConfig in resnet3d_OASIS.py)

@dataclass
class MedicalNetConfig:
    model_depth: int = 18                       # 10, 18, 34, 50 supported by 23-dataset weights
    shortcut_type: str = "A"                    # "A" for resnet18/34, "B" for resnet10/50
    pretrained_path: Optional[str] = None       # path to .pth file from MedicalNet repo
    in_channels: int = 1                        # MRI is single-channel - matches MedicalNet default
    out_dim: int = 3                            # 3 classes: CN, MCI, AD
    dropout: float = 0.2                        # applied before final classifier
    no_cuda: bool = False                       # set True if running CPU-only


# Conv helper (matches MedicalNet's resnet.py)

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        out.size(4),
        device=out.device,
        dtype=out.dtype,
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


# Basic Block (used by ResNet 10/18/34)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# Bottleneck (used by ResNet 50)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, padding=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# Backbone (matches MedicalNet's segmentation backbone exactly)

class MedicalNetBackbone(nn.Module):
    """
    Faithful reproduction of MedicalNet's resnet.py backbone (segmentation variant).
    Using stride 2 for the first conv to match standard 3D ResNet pretraining.
    The classification head is added separately in MedicalNetClassifier below.
    """

    def __init__(self, block, layers, shortcut_type="B", no_cuda=False, in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda

        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=7, stride=(2, 2, 2),
            padding=(3, 3, 3), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x



# Classifier wrapper - adds head matching your existing model interface

class MedicalNetClassifier(nn.Module):
    """
    Wraps MedicalNetBackbone with global average pooling + classification head.
    Forward signature matches torchvision r3d_18 wrapper used in resnet3d_OASIS.py
    so it can be used as a drop-in replacement in train.py.
    """

    def __init__(self, backbone: MedicalNetBackbone, feat_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Named "fc" to match torchvision r3d_18 convention used elsewhere
        self.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(feat_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)        # [B, feat_dim, D', H', W']
        x = self.avgpool(x)         # [B, feat_dim, 1, 1, 1]
        x = torch.flatten(x, 1)     # [B, feat_dim]
        return self.fc(x)


# Depth -> layer config map

_DEPTH_CONFIG = {
    10: (BasicBlock, [1, 1, 1, 1], 512),
    18: (BasicBlock, [2, 2, 2, 2], 512),
    34: (BasicBlock, [3, 4, 6, 3], 512),
    50: (Bottleneck, [3, 4, 6, 3], 2048),
}


# Pretrained weight loader (handles MedicalNet's quirks)

def _load_medicalnet_pretrained(model: MedicalNetClassifier, weights_path: str) -> dict:
    """
    Loads MedicalNet pretrained weights into our model.

    MedicalNet weights have several differences:
      1. Saved as a checkpoint dict with "state_dict" key
      2. All keys prefixed with "module." (DataParallel artefact)
      3. Original model has segmentation head (conv_seg) we don't want
      4. Backbone keys need to be remapped to our backbone's namespace

    Returns a report dict with counts of matched/skipped/missing keys.
    """
    ckpt = torch.load(weights_path, map_location="cpu")

    # Extract state dict from checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip "module." prefix from DataParallel saved weights
    stripped = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        stripped[new_key] = v

    # Remap backbone keys: original was top-level, ours is under "backbone."
    # Skip segmentation head (conv_seg) which we don't use for classification.
    # Skip our classification head ("fc.") - randomly initialised.
    remapped = {}
    skipped_seg = 0
    for k, v in stripped.items():
        if k.startswith("conv_seg"):
            skipped_seg += 1
            continue
        remapped[f"backbone.{k}"] = v

    # Load with strict=False so missing fc head doesn't error
    own_state = model.state_dict()
    matched, shape_mismatch, missing = 0, 0, 0

    for k, v in remapped.items():
        if k in own_state:
            if own_state[k].shape == v.shape:
                own_state[k].copy_(v)
                matched += 1
            else:
                shape_mismatch += 1
        else:
            missing += 1

    return {
        "matched": matched,
        "shape_mismatch": shape_mismatch,
        "missing_in_model": missing,
        "skipped_segmentation_head": skipped_seg,
        "total_pretrained_keys": len(remapped),
    }




# ----------------------------------------------------------------------------
# Public factory function (mirrors build_model in resnet3d_OASIS.py)

def build_medicalnet(cfg: MedicalNetConfig) -> nn.Module:
    """
    Build a MedicalNet 3D ResNet classifier with optional pretrained weights.

    Same interface as build_model in resnet3d_OASIS.py - takes a config object,
    returns an nn.Module ready to be trained.
    """
    depth = int(cfg.model_depth)
    if depth not in _DEPTH_CONFIG:
        raise ValueError(
            f"Unsupported model_depth={depth}. "
            f"Supported: {sorted(_DEPTH_CONFIG.keys())}"
        )

    block, layers, feat_dim = _DEPTH_CONFIG[depth]

    backbone = MedicalNetBackbone(
        block=block,
        layers=layers,
        shortcut_type=str(cfg.shortcut_type).upper(),
        no_cuda=bool(cfg.no_cuda),
        in_channels=int(cfg.in_channels),
    )

    model = MedicalNetClassifier(
        backbone=backbone,
        feat_dim=feat_dim,
        out_dim=int(cfg.out_dim),
        dropout=float(cfg.dropout),
    )

    if cfg.pretrained_path:
        report = _load_medicalnet_pretrained(model, cfg.pretrained_path)
        print(f"[MedicalNet] Loaded pretrained weights from {cfg.pretrained_path}")
        print(f"[MedicalNet] matched: {report['matched']}")
        print(f"[MedicalNet] skipped seg head: {report['skipped_segmentation_head']}")
        print(f"[MedicalNet] shape mismatches: {report['shape_mismatch']}")
        print(f"[MedicalNet] missing in model: {report['missing_in_model']}")
        if report["matched"] == 0:
            raise RuntimeError(
                "No pretrained weights matched the model. "
                "Check that model_depth and shortcut_type match the .pth file."
            )

    return model


# Helpers for freeze/unfreeze (used by train.py)

def freeze_backbone(model: MedicalNetClassifier) -> int:
    """Freeze backbone parameters. Returns count of frozen parameters."""
    n = 0
    for p in model.backbone.parameters():
        p.requires_grad = False
        n += p.numel()
    return n


def unfreeze_all(model: MedicalNetClassifier) -> int:
    """Unfreeze all parameters. Returns count of unfrozen parameters."""
    n = 0
    for p in model.parameters():
        p.requires_grad = True
        n += p.numel()
    return n
