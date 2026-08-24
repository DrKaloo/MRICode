"""
MobileNet3D V2 wrapper for 3D medical volumes, with Kinetics-600 pretraining.

Mirrors the interface of medicalnet3d.py / resnet3d_OASIS.py so it slots into
train_medicalnet.py as a third backbone option alongside r3d_18 (Kinetics-400)
and MedicalNet ResNet-18 (medical 3D pretraining).

Architecture: Köpüklü et al. 2019, "Resource Efficient 3D Convolutional Neural
Networks for Action Recognition." https://arxiv.org/abs/1904.02422
Reference implementation: https://github.com/okankop/Efficient-3DCNNs

Pretrained weights: download `kinetics_mobilenetv2_1.0x_RGB_16_best.pth` (or
similar width variant) from the Köpüklü repo's Google Drive link, place under
`pretrained/`, and pass via --mobilenet3d_weights.

Architectural note on first-layer stride:
The reference impl uses stride (1, 2, 2) on the first conv -- temporal stride
1, spatial stride 2 -- because Kinetics-600 is video where the first axis is
time. For MRI, where all three axes are spatial, this gives mildly asymmetric
downsampling along the first axis. We preserve it because changing it would
invalidate the pretrained weights downstream. The asymmetry is harmless at
128^3 resolution: final feature shape becomes (8, 4, 4) instead of (4, 4, 4),
which is then collapsed by the global avg pool.

Usage:
    from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2
    cfg = MobileNet3DV2Config(
        width_mult=1.0,
        pretrained_path="pretrained/kinetics_mobilenetv2_1.0x_RGB_16_best.pth",
    )
    model = build_mobilenet3d_v2(cfg)
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn

# Reuse the channel-adapter from resnet3d_OASIS so 3->1 conversion is
# bit-identical across all our pretrained backbones (mean across input channels)
from resnet3d_OASIS import _replace_first_conv3d


# Configuration dataclass (mirrors MedicalNetConfig)

@dataclass
class MobileNet3DV2Config:
    width_mult: float = 1.0                    # 0.2/0.45/0.7/1.0/1.4 supported by Köpüklü repo weights
    pretrained_path: Optional[str] = None      # path to .pth from Efficient-3DCNNs repo
    in_channels: int = 1                       # 1 for grayscale MRI (matches OASIS pipeline)
    out_dim: int = 3                           # CN / MCI / AD
    dropout: float = 0.2                       # applied before final classifier


# Building blocks (parallel to Köpüklü's mobilenetv2.py)

def _conv_bn(inp: int, oup: int, stride) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


def _conv_1x1x1_bn(inp: int, oup: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    """3D inverted residual block: pw expand -> dw 3x3x3 -> pw-linear project."""

    def __init__(self, inp: int, oup: int, stride, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (self.stride == (1, 1, 1)) and (inp == oup)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise expand
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


# Backbone (matches Köpüklü's MobileNetV2 features, no classifier)

class MobileNet3DV2Backbone(nn.Module):
    """
    Based on the reproduction of the feature extractor from Köpüklü's MobileNetV2.
    The classifier head is added separately in MobileNet3DV2Classifier below.
    """

    # t = expansion ratio, c = output channels, n = num blocks, s = stride
    INVERTED_RESIDUAL_SETTING = [
        [1,  16, 1, (1, 1, 1)],
        [6,  24, 2, (2, 2, 2)],
        [6,  32, 3, (2, 2, 2)],
        [6,  64, 4, (2, 2, 2)],
        [6,  96, 3, (1, 1, 1)],
        [6, 160, 3, (2, 2, 2)],
        [6, 320, 1, (1, 1, 1)],
    ]

    def __init__(self, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()

        input_channel = int(32 * width_mult)
        # MobileNetV2 convention: NOT to shrink last_channel below 1280 even if width_mult < 1
        self.last_channel = int(1280 * max(1.0, width_mult))

        layers = [_conv_bn(in_channels, input_channel, stride=(1, 2, 2))]

        for t, c, n, s in self.INVERTED_RESIDUAL_SETTING:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel

        layers.append(_conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# Classifier wrapper (mirrors MedicalNetClassifier)

class MobileNet3DV2Classifier(nn.Module):
    """
    Backbone + global avgpool + classification head.

    The head is named `fc` to match the convention in resnet3d_OASIS.py and
    medicalnet3d.py, so train_medicalnet.py's name-based freeze_backbone
    (which checks for `.fc.` in parameter names) works without modification.
    """

    def __init__(
        self,
        backbone: MobileNet3DV2Backbone,
        last_channel: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(last_channel, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)         # [B, last_channel, D', H', W']
        x = self.avgpool(x)          # [B, last_channel, 1, 1, 1]
        x = torch.flatten(x, 1)      # [B, last_channel]
        return self.fc(x)


# Pretrained weight loader (to handle Köpüklü's differences)

def _load_mobilenet3d_pretrained(
    model: MobileNet3DV2Classifier, weights_path: str
) -> dict:
    """
    Loads Köpüklü Efficient-3DCNNs pretrained weights into our model.

    Differences (parallel to the MedicalNet loader):
      1. Saved as a checkpoint dict with "state_dict" key
      2. Keys prefixed with "module." (DataParallel artefact)
      3. Original has Kinetics-600 classifier head we don't want
      4. Original "features.*" keys need remapping to "backbone.features.*"

    Returns a report dict with counts of matched / skipped / missing keys.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip "module." prefix from DataParallel-saved weights
    stripped = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        stripped[new_key] = v

    # Skip the Kinetics-600 classifier head; remap features into our backbone namespace
    remapped = {}
    skipped_classifier = 0
    skipped_unknown = 0
    for k, v in stripped.items():
        if k.startswith("classifier"):
            skipped_classifier += 1
            continue
        if k.startswith("features."):
            remapped[f"backbone.{k}"] = v
        else:
            skipped_unknown += 1

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
        "skipped_classifier_head": skipped_classifier,
        "skipped_unknown_keys": skipped_unknown,
        "total_pretrained_keys": len(remapped),
    }


# Public factory function (mirrors build_medicalnet)

def build_mobilenet3d_v2(cfg: MobileNet3DV2Config) -> nn.Module:
    """
    Build a MobileNet3D V2 classifier with optional Kinetics-600 pretrained weights.

    Order of operations matters:
      1. Build backbone with in_channels=3 to match pretrained weight shape
      2. Load pretrained weights (3-channel first conv loads cleanly)
      3. Adapt first conv to target in_channels via mean-across-channels
         (using the same _replace_first_conv3d as resnet3d_OASIS for consistency)
    """
    width_mult = float(cfg.width_mult)

    # Build with 3 input channels initially so the pretrained first conv loads.
    backbone = MobileNet3DV2Backbone(in_channels=3, width_mult=width_mult)
    last_channel = backbone.last_channel

    model = MobileNet3DV2Classifier(
        backbone=backbone,
        last_channel=last_channel,
        out_dim=int(cfg.out_dim),
        dropout=float(cfg.dropout),
    )

    if cfg.pretrained_path:
        report = _load_mobilenet3d_pretrained(model, cfg.pretrained_path)
        print(f"[MobileNet3DV2] Loaded pretrained weights from {cfg.pretrained_path}")
        print(f"[MobileNet3DV2] matched: {report['matched']}")
        print(f"[MobileNet3DV2] skipped classifier head: {report['skipped_classifier_head']}")
        print(f"[MobileNet3DV2] skipped unknown keys: {report['skipped_unknown_keys']}")
        print(f"[MobileNet3DV2] shape mismatches: {report['shape_mismatch']}")
        print(f"[MobileNet3DV2] missing in model: {report['missing_in_model']}")
        if report["matched"] == 0:
            raise RuntimeError(
                "No pretrained weights matched the model. "
                f"Check that width_mult={width_mult} corresponds to the .pth file "
                "(e.g. 1.0x weights need width_mult=1.0)."
            )

    # Adapt first conv to the requested in_channels (1 for MRI -> mean across 3->1)
    if int(cfg.in_channels) != 3:
        _replace_first_conv3d(
            model,
            in_channels=int(cfg.in_channels),
            pretrained=bool(cfg.pretrained_path),
        )

    return model


# Helpers (mirror medicalnet3d.py API; train_medicalnet.py mirrors these
# with a name-based version, but kept here for parity)

def freeze_backbone(model: MobileNet3DV2Classifier) -> int:
    """Freeze backbone parameters. Returns count of frozen parameters."""
    n = 0
    for p in model.backbone.parameters():
        p.requires_grad = False
        n += p.numel()
    return n


def unfreeze_all(model: MobileNet3DV2Classifier) -> int:
    """Unfreeze all parameters. Returns count of unfrozen parameters."""
    n = 0
    for p in model.parameters():
        p.requires_grad = True
        n += p.numel()
    return n
