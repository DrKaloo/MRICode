from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    backbone: str = "r3d_18"  # r3d_18, r2plus1d_18, custom_small imported
    pretrained: bool = False
    in_channels: int = 1
    out_dim: int = 3
    dropout: float = 0.2


def _set_module_by_name(root: nn.Module, name: str, new_module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _replace_first_conv3d(model: nn.Module, in_channels: int, pretrained: bool) -> None:
    first_name = None
    first_conv = None

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            first_name = name
            first_conv = m
            break

    if first_conv is None or first_name is None:
        return

    if first_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv3d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        dilation=first_conv.dilation,
        groups=first_conv.groups if first_conv.groups == 1 else 1,
        bias=(first_conv.bias is not None),
        padding_mode=first_conv.padding_mode,
    )

    with torch.no_grad():
        if pretrained and first_conv.weight is not None and first_conv.weight.ndim == 5:
            w = first_conv.weight  # (out, in, kD, kH, kW)
            if w.shape[1] == 3 and in_channels == 1:
                new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
            elif w.shape[1] == 3 and in_channels > 1:
                base = w.mean(dim=1, keepdim=True)  # [out,1,...]
                new_conv.weight.copy_(base.repeat(1, in_channels, 1, 1, 1) / float(in_channels))
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

    _set_module_by_name(model, first_name, new_conv)


class CustomSmall3D(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, dropout: float):
        super().__init__()

        def block(cin, cout, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            block(in_channels, 32, 3, 1, 1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            block(32, 64, 3, 1, 1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            block(64, 128, 3, 1, 1),
            nn.MaxPool3d(kernel_size=2, stride=2),
            block(128, 256, 3, 1, 1),
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def build_model(cfg: ModelConfig) -> nn.Module:
    backbone = str(cfg.backbone).lower().strip()
    in_ch = int(cfg.in_channels)
    out_dim = int(cfg.out_dim)
    dropout = float(cfg.dropout)

    if backbone == "custom_small":
        return CustomSmall3D(in_channels=in_ch, out_dim=out_dim, dropout=dropout)

    # torchvision video backbones
    try:
        from torchvision.models.video import r3d_18, r2plus1d_18
        from torchvision.models.video import R3D_18_Weights, R2Plus1D_18_Weights
    except Exception as e:
        raise RuntimeError(
            "torchvision video models not available. Install torchvision according to graphics version"
        ) from e            #  download and get to use GPU -> therefore warning and version

    # Imported Backbones
    if backbone == "r3d_18":
        weights = R3D_18_Weights.KINETICS400_V1 if cfg.pretrained else None
        model = r3d_18(weights=weights)
    elif backbone == "r2plus1d_18":
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if cfg.pretrained else None
        model = r2plus1d_18(weights=weights)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    _replace_first_conv3d(model, in_channels=in_ch, pretrained=bool(cfg.pretrained))

    # Replace classifier head to out_dim
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, out_dim),
    )
    return model