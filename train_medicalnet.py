import os
import json
import time
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import BrainMRIDataset
from resnet3d_OASIS import ModelConfig, build_model
from medicalnet3d import MedicalNetConfig, build_medicalnet
from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2

try:
    from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
except Exception:
    confusion_matrix = roc_auc_score = average_precision_score = None

BASE_DIR = Path(__file__).resolve().parent



# Auto splits_dir

def _looks_like_validated_splits(p: Path) -> bool:
    return p.is_dir() and (p / "train.csv").is_file() and (p / "val.csv").is_file()


def auto_find_splits_dir() -> Path:
    candidates = []
    roots = [BASE_DIR / "data" / "splits", BASE_DIR / "splits"]

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("validated"):
            if _looks_like_validated_splits(p):
                candidates.append(p)
        for p in root.rglob("*"):
            if _looks_like_validated_splits(p):
                candidates.append(p)

    if not candidates:
        raise SystemExit(
            "No valid splits directory found.\n"
            "Expected a folder containing train.csv and val.csv under:\n"
            f"  {BASE_DIR / 'data' / 'splits'}\n"
            f"  {BASE_DIR / 'splits'}\n"
            "Fix: run data_splits.py first, or pass --splits_dir explicitly."
        )

    candidates = sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_splits_dir(arg_value) -> str:
    if arg_value is None or str(arg_value).strip() == "":
        return str(auto_find_splits_dir())

    p = Path(str(arg_value)).expanduser()
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()

    if not _looks_like_validated_splits(p):
        raise SystemExit(
            f"--splits_dir does not look valid: {p}\n"
            "Expected files: n"
            f"  {p / 'train.csv'}\n"
            f"  {p / 'val.csv'}"
        )

    return str(p)



# Repro

def set_seed(seed: int, deterministic: bool = False) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False



# Transforms

class Compose:
    def __init__(self, ops):
        self.ops = list(ops)

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x


class RandomCrop3D:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,D,H,W]
        _, D, H, W = x.shape
        s = self.size
        if s >= D and s >= H and s >= W:
            return x
        d0 = 0 if D == s else torch.randint(0, D - s + 1, (1,)).item()
        h0 = 0 if H == s else torch.randint(0, H - s + 1, (1,)).item()
        w0 = 0 if W == s else torch.randint(0, W - s + 1, (1,)).item()
        return x[:, d0 : d0 + s, h0 : h0 + s, w0 : w0 + s]


class CenterCrop3D:
    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, D, H, W = x.shape
        s = self.size
        if s >= D and s >= H and s >= W:
            return x
        d0 = (D - s) // 2
        h0 = (H - s) // 2
        w0 = (W - s) // 2
        return x[:, d0 : d0 + s, h0 : h0 + s, w0 : w0 + s]


class RandomFlipW3D:
    """
    Single-axis flip only (W axis). Avoids unphysical flips across multiple axes
    """
    def __init__(self, p: float = 0.5):
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: [C,D,H,W]
        if torch.rand(1).item() < self.p:
            x = torch.flip(x, dims=[3])
        return x


class AdditiveGaussianNoise:
    def __init__(self, sigma: float = 0.02, p: float = 0.3):
        self.sigma = float(sigma)
        self.p = float(p)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return x
        return x + torch.randn_like(x) * self.sigma



# Dataloader: robust collate (keeps meta as list) -------------------> ATTENTION FOR LATER

def collate_keep_meta(batch: List[dict]) -> Dict[str, object]:
    xs = []
    ys = []
    metas = []
    has_meta = ("meta" in batch[0])

    for b in batch:
        x = b["x"]
        y = b["y"]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        xs.append(x)
        ys.append(int(y))
        if has_meta:
            metas.append(b.get("meta", None))

    out: Dict[str, object] = {
        "x": torch.stack(xs, dim=0),
        "y": torch.tensor(ys, dtype=torch.long),
    }
    if has_meta:
        out["meta"] = metas
    return out



# Metrics

def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    if confusion_matrix is not None:
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)          # true counts per class
    pred_sum = cm.sum(axis=0).astype(np.float64)         # predicted counts per class

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    acc = float(tp.sum() / max(1.0, cm.sum()))
    bal_acc = float(np.mean(recall)) if recall.size else float("nan")
    macro_f1 = float(np.mean(f1)) if f1.size else float("nan")
    macro_prec = float(np.mean(precision)) if precision.size else float("nan")
    macro_rec = float(np.mean(recall)) if recall.size else float("nan")

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


def safe_auc_pr_multiclass(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> Tuple[float, float]:
    if roc_auc_score is None or average_precision_score is None:
        return float("nan"), float("nan")
    y_true = y_true.astype(int)
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        return float("nan"), float("nan")
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    y_onehot = np.eye(num_classes, dtype=np.int32)[y_true]
    try:
        roc = float(roc_auc_score(y_onehot, probs, average="macro", multi_class="ovr"))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_onehot, probs, average="macro"))
    except Exception:
        pr = float("nan")
    return roc, pr



# Imbalance helpers

def infer_num_classes(df: pd.DataFrame) -> int:
    y = df["label"].astype(int).to_numpy()
    return int(np.max(y)) + 1


def make_balanced_sampler_multiclass(labels: np.ndarray, num_classes: int) -> WeightedRandomSampler:
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    sample_w = torch.as_tensor(inv[labels], dtype=torch.double)
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    total = counts.sum()
    w = total / (num_classes * counts)
    return w.astype(np.float32)



# AMP

def get_amp():
    try:
        return torch.amp.autocast, torch.amp.GradScaler
    except Exception:
        return torch.cuda.amp.autocast, torch.cuda.amp.GradScaler



# Freezing Backbone epochs

def freeze_backbone(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if ".fc." in name or name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# Val TTA (9-crop from full volume)

def make_9crop_indices(D: int, H: int, W: int, patch: int) -> List[Tuple[int, int, int]]:
    d0, h0, w0 = 0, 0, 0
    d1, h1, w1 = max(0, D - patch), max(0, H - patch), max(0, W - patch)
    dc, hc, wc = max(0, (D - patch) // 2), max(0, (H - patch) // 2), max(0, (W - patch) // 2)

    corners = [
        (d0, h0, w0),
        (d0, h0, w1),
        (d0, h1, w0),
        (d0, h1, w1),
        (d1, h0, w0),
        (d1, h0, w1),
        (d1, h1, w0),
        (d1, h1, w1),
    ]
    center = [(dc, hc, wc)]
    return corners + center


def forward_tta_9crop(model: nn.Module, x_full: torch.Tensor, patch: int) -> torch.Tensor:
    # x_full: [B,1,D,H,W] -> returns logits [B,C] averaged over 9 crops
    B, C, D, H, W = x_full.shape
    idx = make_9crop_indices(D, H, W, patch)
    crops = []
    for (d, h, w) in idx:
        crops.append(x_full[:, :, d : d + patch, h : h + patch, w : w + patch])
    x = torch.stack(crops, dim=1)  # [B,T,1,p,p,p]
    B, T, C1, pD, pH, pW = x.shape
    x = x.view(B * T, C1, pD, pH, pW)
    logits = model(x)  # [B*T, num_classes]
    logits = logits.view(B, T, -1).mean(dim=1)  # [B, num_classes]
    return logits



# EMA

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        sd = model.state_dict()
        for k, v in sd.items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> dict:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        new_sd = backup.copy()
        for k, v in self.shadow.items():
            new_sd[k] = v.detach().clone()
        model.load_state_dict(new_sd, strict=False)
        return backup

    def restore(self, model: nn.Module, backup: dict):
        model.load_state_dict(backup, strict=False)



# Loss: Focal + optional smoothing + optional class weights

class FocalCrossEntropyLoss(nn.Module):
    """
    Multiclass focal CE with optional class weights and label smoothing.
    - gamma <= 0 -> behaves like CE (with smoothing/weights)
    """
    def __init__(
        self,
        gamma: float = 1.5,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.label_smoothing = float(label_smoothing)
        self.reduction = str(reduction)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: [B,C], target: [B]
        log_probs = F.log_softmax(logits, dim=1)  # [B,C]
        probs = log_probs.exp()

        # NLL for true class
        nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # [B]

        # Label smoothing: blend with uniform loss
        if self.label_smoothing > 0.0:
            smooth = -log_probs.mean(dim=1)  # [B]
            nll = (1.0 - self.label_smoothing) * nll + self.label_smoothing * smooth

        # Class weights
        if self.weight is not None:
            w = self.weight.gather(0, target)  # [B]
            nll = nll * w

        # Focal factor using p_t from *true class*
        if self.gamma > 0.0:
            p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
            focal = (1.0 - p_t).pow(self.gamma)
            nll = nll * focal

        if self.reduction == "sum":
            return nll.sum()
        if self.reduction == "none":
            return nll
        return nll.mean()



# Args

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default=None)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--deterministic", action="store_true")

    ap.add_argument("--backbone", default="r3d_18",
                    choices=["r3d_18", "r2plus1d_18", "custom_small", "medicalnet_resnet18", "mobilenet3d_v2"])
    ap.add_argument("--medicalnet_weights", default=None,
                    help="Path to MedicalNet pretrained .pth file (e.g. pretrained/resnet_18_23dataset.pth)")
    ap.add_argument("--mobilenet3d_weights", default=None,
                    help="Path to Köpüklü MobileNet3DV2 .pth (e.g. pretrained/kinetics_mobilenetv2_1.0x_RGB_16_best.pth)")
    ap.add_argument("--mobilenet3d_width_mult", type=float, default=1.0)

    # Default pretrained ON unless --no_pretrained
    ap.add_argument("--pretrained", action="store_true", help="Force pretrained ON (default already ON)")
    ap.add_argument("--no_pretrained", action="store_true", help="Disable pretrained weights")

    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)

    # Defaults tuned for stability
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--patch_size", type=int, default=96)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--cache_size", type=int, default=0)

    ap.add_argument("--balanced_sampler", action="store_true", help="Do not combine with class weights; auto-disables weights if enabled")
    ap.add_argument("--no_class_weights", action="store_true")

    ap.add_argument("--freeze_epochs", type=int, default=10)
    ap.add_argument("--early_patience", type=int, default=12)
    ap.add_argument("--monitor", default="macro_f1", choices=["macro_f1", "bal_acc", "acc", "val_loss"])

    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--focal_gamma", type=float, default=1.5)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--val_tta_9crop", action="store_true")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)

    return ap



# Main

def main():
    args = build_argparser().parse_args()
    args.splits_dir = resolve_splits_dir(args.splits_dir)

    set_seed(int(args.seed), deterministic=bool(args.deterministic))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv = os.path.join(args.splits_dir, "val.csv")
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise RuntimeError(f"Missing train/val CSVs in: {args.splits_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    num_classes = infer_num_classes(train_df)
    if num_classes < 2:
        raise RuntimeError(f"Invalid num_classes inferred: {num_classes}")

    print(f"[Splits] {args.splits_dir}")
    print(f"Device: {device.type}")
    print(f"Classes: {num_classes}")
    print(f"TRAIN   subjects: {train_df['subject_id'].nunique()}")
    print(f"VAL     subjects: {val_df['subject_id'].nunique()}")
    print("TRAIN    label counts:\n", train_df["label"].value_counts().sort_index().to_string())
    print("VAL      label counts:\n", val_df["label"].value_counts().sort_index().to_string())

    patch = int(args.patch_size)
    print(f"Patch: {patch}^3 (train random crop; val {'9-crop TTA' if args.val_tta_9crop else 'center crop'})")

    # Safer augmentation
    train_tf = Compose([
        RandomCrop3D(patch),
        RandomFlipW3D(0.5),
        AdditiveGaussianNoise(0.02, 0.3),
    ])

    # If there is use of val TTA, do NOT crop in dataset; crop inside val loop from full 128^3 res
    val_tf = None if bool(args.val_tta_9crop) else Compose([CenterCrop3D(patch)])

    ds_train = BrainMRIDataset(train_csv, transform=train_tf, cache_size=int(args.cache_size))
    ds_val = BrainMRIDataset(val_csv, transform=val_tf, cache_size=0)

    y_train = train_df["label"].astype(int).to_numpy()

    # Enforce: do not double-balance
    if bool(args.balanced_sampler) and (not bool(args.no_class_weights)):
        args.no_class_weights = True
        print("[INFO] balanced_sampler enabled -> disabling class weights to avoid double-balancing.")

    sampler = None
    shuffle = True
    if args.balanced_sampler:
        sampler = make_balanced_sampler_multiclass(y_train, num_classes=num_classes)
        shuffle = False

    pin_memory = bool(args.pin_memory) and (device.type == "cuda")

    loader_train = DataLoader(
        ds_train,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_keep_meta,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_keep_meta,
    )

    # Default pretrained ON unless set: disabled
    use_pretrained = True
    if bool(args.no_pretrained):
        use_pretrained = False
    if bool(args.pretrained):
        use_pretrained = True

    if str(args.backbone) == "medicalnet_resnet18":
        med_cfg = MedicalNetConfig(
            model_depth=18,
            shortcut_type="A",
            pretrained_path=str(args.medicalnet_weights) if args.medicalnet_weights else None,
            in_channels=1,
            out_dim=num_classes,
            dropout=float(args.dropout),
            no_cuda=(device.type != "cuda"),
        )
        cfg = med_cfg
        model = build_medicalnet(med_cfg).to(device)
    elif str(args.backbone) == "mobilenet3d_v2":
        mob_cfg = MobileNet3DV2Config(
            width_mult=float(args.mobilenet3d_width_mult),
            pretrained_path=str(args.mobilenet3d_weights) if args.mobilenet3d_weights else None,
            in_channels=1,
            out_dim=num_classes,
            dropout=float(args.dropout),
        )
        cfg = mob_cfg
        model = build_mobilenet3d_v2(mob_cfg).to(device)
    else:
        cfg = ModelConfig(
            backbone=str(args.backbone),
            pretrained=use_pretrained,
            in_channels=1,
            out_dim=num_classes,
            dropout=float(args.dropout),
        )
        model = build_model(cfg).to(device)

    if int(args.freeze_epochs) > 0 and (
            use_pretrained or str(args.backbone) in {"medicalnet_resnet18", "mobilenet3d_v2"}):
        freeze_backbone(model)

    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"pretrained: {use_pretrained} | freeze_epochs: {int(args.freeze_epochs)}")

    # Class weights
    class_w = None
    if not bool(args.no_class_weights):
        w_np = compute_class_weights(y_train, num_classes=num_classes)
        class_w = torch.tensor(w_np, dtype=torch.float32, device=device)
        print(f"class_weights: {w_np.round(4).tolist()}")

    # Loss: focal CE + optional smoothing
    loss_fn = FocalCrossEntropyLoss(
        gamma=float(args.focal_gamma),
        weight=class_w,
        label_smoothing=max(0.0, float(args.label_smoothing)),
        reduction="mean",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs), eta_min=float(args.lr) * 0.05
    )

    autocast, GradScaler = get_amp()
    scaler = None
    if device.type == "cuda":
        try:
            scaler = GradScaler("cuda", enabled=True)
        except TypeError:
            scaler = GradScaler(enabled=True)

    ema = EMA(model, decay=float(args.ema_decay)) if bool(args.ema) else None

    best_score = -1e18
    best_path = os.path.join(args.results_dir, "best.pt")
    epochs_no_improve = 0

    def pick_monitor(m: Dict[str, float], val_loss: float) -> float:
        key = str(args.monitor)
        if key == "val_loss":
            return -float(val_loss)
        return float(m.get(key, float("nan")))

    for epoch in range(1, int(args.epochs) + 1):
        # Unfreeze stage: lower LR a bit
        if int(args.freeze_epochs) > 0 and (
                use_pretrained or str(args.backbone) in {"medicalnet_resnet18", "mobilenet3d_v2"}) and epoch == int(
                args.freeze_epochs) + 1:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(args.lr) * 0.5,
                weight_decay=float(args.weight_decay),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(args.epochs) - epoch + 1),
                eta_min=float(args.lr) * 0.05,
            )

        model.train()
        t0 = time.time()
        running_loss = 0.0

        tr_y = []
        tr_pred = []
        tr_probs = []

        optimizer.zero_grad(set_to_none=True)
        accum = max(1, int(args.accum_steps))
        opt_steps = 0

        for step, batch in enumerate(tqdm(loader_train, desc=f"Train E{epoch}", leave=False)):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y) / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item()) * accum

            if (step + 1) % accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

                if ema is not None:
                    ema.update(model)

            with torch.no_grad():
                probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy()
                pred = probs.argmax(axis=1)
                tr_probs.append(probs)
                tr_pred.append(pred)
                tr_y.append(y.detach().cpu().numpy())

        if (len(loader_train) % accum) != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

            if ema is not None:
                ema.update(model)

        scheduler.step()

        tr_y_np = np.concatenate(tr_y) if tr_y else np.array([], dtype=int)
        tr_pred_np = np.concatenate(tr_pred) if tr_pred else np.array([], dtype=int)
        tr_probs_np = np.concatenate(tr_probs) if tr_probs else np.zeros((0, num_classes), dtype=np.float32)

        tr_cm = _confusion_matrix(tr_y_np, tr_pred_np, num_classes=num_classes)
        tr_m = metrics_from_cm(tr_cm)
        tr_roc, tr_pr = safe_auc_pr_multiclass(tr_y_np, tr_probs_np, num_classes=num_classes)

        # --- Validation ---
        def run_val(eval_model: nn.Module) -> Tuple[float, Dict[str, float], float, float, np.ndarray]:
            eval_model.eval()
            val_loss_sum = 0.0
            va_y = []
            va_pred = []
            va_probs = []

            with torch.no_grad():
                for batch in tqdm(loader_val, desc=f"Val E{epoch}", leave=False):
                    x_full = batch["x"].to(device, non_blocking=True)
                    yv = batch["y"].to(device, non_blocking=True)

                    with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                        if bool(args.val_tta_9crop):
                            logits = forward_tta_9crop(eval_model, x_full, patch=patch)
                        else:
                            logits = eval_model(x_full)
                        loss = loss_fn(logits, yv)

                    val_loss_sum += float(loss.item())
                    probs = torch.softmax(logits, dim=1).float().cpu().numpy()
                    pred = probs.argmax(axis=1)

                    va_probs.append(probs)
                    va_pred.append(pred)
                    va_y.append(yv.cpu().numpy())

            va_y_np = np.concatenate(va_y) if va_y else np.array([], dtype=int)
            va_pred_np = np.concatenate(va_pred) if va_pred else np.array([], dtype=int)
            va_probs_np = np.concatenate(va_probs) if va_probs else np.zeros((0, num_classes), dtype=np.float32)

            va_cm = _confusion_matrix(va_y_np, va_pred_np, num_classes=num_classes)
            va_m = metrics_from_cm(va_cm)
            va_roc, va_pr = safe_auc_pr_multiclass(va_y_np, va_probs_np, num_classes=num_classes)
            val_loss_mean = val_loss_sum / max(1, len(loader_val))
            return val_loss_mean, va_m, va_roc, va_pr, va_cm

        val_loss_mean, va_m, va_roc, va_pr, va_cm = run_val(model)

        ema_used = False
        if ema is not None:
            backup = ema.apply_to(model)
            val_loss_ema, va_m_ema, va_roc_ema, va_pr_ema, va_cm_ema = run_val(model)
            ema.restore(model, backup)

            score_raw = pick_monitor(va_m, val_loss_mean)
            score_ema = pick_monitor(va_m_ema, val_loss_ema)
            if score_ema >= score_raw:
                val_loss_mean, va_m, va_roc, va_pr, va_cm = val_loss_ema, va_m_ema, va_roc_ema, va_pr_ema, va_cm_ema
                ema_used = True

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{int(args.epochs)} | "
            f"train_loss {running_loss/max(1,len(loader_train)):.4f} | "
            f"train acc {tr_m['acc']:.3f} bal {tr_m['bal_acc']:.3f} f1 {tr_m['macro_f1']:.3f} | "
            f"train ROC {tr_roc:.3f} PR {tr_pr:.3f} | "
            f"val_loss {val_loss_mean:.4f} | "
            f"val acc {va_m['acc']:.3f} bal {va_m['bal_acc']:.3f} f1 {va_m['macro_f1']:.3f} | "
            f"val ROC {va_roc:.3f} PR {va_pr:.3f} | "
            f"{'EMA' if ema_used else 'RAW'} | "
            f"opt_steps {opt_steps} | {dt:.1f}s"
        )

        score = pick_monitor(va_m, val_loss_mean)
        improved = score > best_score

        if improved:
            best_score = float(score)
            epochs_no_improve = 0

            ckpt = {
                "model_state": model.state_dict(),
                "cfg": asdict(cfg),
                "epoch": epoch,
                "monitor": str(args.monitor),
                "best_score": best_score,
                "num_classes": num_classes,
                "val_metrics": {**va_m, "roc_auc_macro_ovr": float(va_roc), "pr_auc_macro": float(va_pr)},
                "val_loss": float(val_loss_mean),
                "val_cm": va_cm.tolist(),
                "splits_dir": os.path.abspath(args.splits_dir),
                "args": vars(args),
            }
            if ema is not None:
                ckpt["ema_shadow"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}

            torch.save(ckpt, best_path)
            print(f"Saved Best -> {best_path} ({args.monitor}={best_score:.5f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= int(args.early_patience):
                print(f"Early stop: no improvement for {int(args.early_patience)} epochs.")
                break

    summary = {
        "best_path": os.path.abspath(best_path),
        "best_score": best_score,
        "monitor": str(args.monitor),
        "cfg": asdict(cfg),
        "splits_dir": os.path.abspath(args.splits_dir),
        "args": vars(args),
    }
    with open(os.path.join(args.results_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Best {args.monitor}={best_score:.5f}")


if __name__ == "__main__":
    main()