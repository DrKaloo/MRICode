#eavy attention on Clinical data and training by end of March
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_ADNI import BrainMRIDataset
from resnet3d import ModelConfig, build_model

try:
    from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
except Exception:
    confusion_matrix = roc_auc_score = average_precision_score = None

BASE_DIR = Path(__file__).resolve().parent


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
    if not candidates:
        raise SystemExit("No validated splits found (train.csv + val.csv).")
    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def resolve_splits_dir(arg_value) -> str:
    if arg_value is None or str(arg_value).strip() == "":
        return str(auto_find_splits_dir())
    p = Path(arg_value).expanduser()
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()
    if not _looks_like_validated_splits(p):
        raise SystemExit(f"splits_dir invalid: {p}")
    return str(p)


def set_seed(seed: int, deterministic: bool = False):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Compose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, x):
        for op in self.ops: x = op(x)
        return x


class RandomCrop3D:
    def __init__(self, s): self.s = s
    def __call__(self, x):
        _, D, H, W = x.shape
        s = self.s
        if s >= D: return x
        d = torch.randint(0, D - s + 1, (1,)).item()
        h = torch.randint(0, H - s + 1, (1,)).item()
        w = torch.randint(0, W - s + 1, (1,)).item()
        return x[:, d:d+s, h:h+s, w:w+s]


class CenterCrop3D:
    def __init__(self, s): self.s = s
    def __call__(self, x):
        _, D, H, W = x.shape
        s = self.s
        d = (D - s) // 2
        h = (H - s) // 2
        w = (W - s) // 2
        return x[:, d:d+s, h:h+s, w:w+s]


def infer_num_classes(df: pd.DataFrame) -> int:
    return int(df["label"].max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default=None)
    ap.add_argument("--results_dir", default="results_adni")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--patch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    args.splits_dir = resolve_splits_dir(args.splits_dir)
    set_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(args.splits_dir, "train.csv")
    val_csv = os.path.join(args.splits_dir, "val.csv")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    num_classes = infer_num_classes(train_df)

    train_ds = BrainMRIDataset(
        train_csv,
        transform=Compose([RandomCrop3D(args.patch_size)])
    )
    val_ds = BrainMRIDataset(
        val_csv,
        transform=Compose([CenterCrop3D(args.patch_size)])
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(ModelConfig(
        backbone="r3d_18",
        pretrained=True,
        in_channels=1,
        out_dim=num_classes
    )).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(args.results_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in train_loader:
            x, y = b["x"].to(device), b["y"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        print(f"Epoch {epoch} done")

    torch.save(model.state_dict(), os.path.join(args.results_dir, "final.pt"))


if __name__ == "__main__":
    main()
