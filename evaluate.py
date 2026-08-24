import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import BrainMRIDataset
from resnet3d_OASIS import ModelConfig, build_model

try:
    from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
except Exception:
    confusion_matrix = roc_auc_score = average_precision_score = None

BASE_DIR = Path(__file__).resolve().parent


def _looks_like_validated_splits(p: Path) -> bool:
    return p.is_dir() and (p / "test.csv").is_file()


def auto_find_test_dir() -> Path:
    roots = [BASE_DIR / "data" / "splits", BASE_DIR / "splits"]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("validated"):
            if _looks_like_validated_splits(p):
                candidates.append(p)
    if not candidates:
        raise SystemExit("No validated test.csv found under data/splits/.../validated or splits/.../validated")
    candidates = sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


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
    support = cm.sum(axis=1).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

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
        return x[:, d0:d0 + s, h0:h0 + s, w0:w0 + s]


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Path to best.pt (default: results/best.pt)")
    ap.add_argument("--splits_dir", default=None, help="validated dir containing test.csv (auto if omitted)")
    ap.add_argument("--results_dir", default="results", help="Where to write evaluation outputs")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--patch_size", type=int, default=96)
    return ap


def main():
    args = build_argparser().parse_args()

    splits_dir = Path(args.splits_dir).expanduser() if args.splits_dir else auto_find_test_dir()
    test_csv = splits_dir / "test.csv"
    if not test_csv.is_file():
        raise RuntimeError(f"Missing: {test_csv}")

    ckpt_path = Path(args.ckpt) if args.ckpt else (Path(args.results_dir) / "best.pt")
    if not ckpt_path.is_file():
        raise RuntimeError(f"Missing checkpoint: {ckpt_path}")

    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = bool(args.pin_memory) and (device.type == "cuda")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_d = ckpt.get("cfg", {})
    num_classes = int(ckpt.get("num_classes", cfg_d.get("out_dim", 3)))

    cfg = ModelConfig(
        backbone=str(cfg_d.get("backbone", "r3d_18")),
        pretrained=False,  # never load pretrained here; we load checkpoint weights
        in_channels=int(cfg_d.get("in_channels", 1)),
        out_dim=num_classes,
        dropout=float(cfg_d.get("dropout", 0.2)),
    )

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv)
    y_true = df["label"].astype(int).to_numpy()

    patch = int(args.patch_size)
    tf = CenterCrop3D(patch)

    ds = BrainMRIDataset(str(test_csv), transform=tf, cache_size=0)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )

    all_probs = []
    all_pred = []
    all_meta = []
    all_y = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = torch.as_tensor(batch["y"], dtype=torch.long).cpu().numpy()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()
            pred = probs.argmax(axis=1)

            all_probs.append(probs)
            all_pred.append(pred)
            all_y.append(y)
            all_meta.extend(batch.get("meta", []))

    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
    pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
    y = np.concatenate(all_y) if all_y else np.array([], dtype=int)

    cm = _confusion_matrix(y, pred, num_classes=num_classes)
    m = metrics_from_cm(cm)
    roc, pr = safe_auc_pr_multiclass(y, probs, num_classes=num_classes)

    out = {
        "ckpt": str(ckpt_path.resolve()),
        "splits_dir": str(splits_dir.resolve()),
        "num_classes": num_classes,
        "metrics": {**m, "roc_auc_macro_ovr": float(roc), "pr_auc_macro": float(pr)},
        "confusion_matrix": cm.tolist(),
    }

    out_json = Path(args.results_dir) / "evaluation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # write per-sample predictions
    out_df = df.copy()
    out_df["pred"] = pred
    for c in range(num_classes):
        out_df[f"prob_{c}"] = probs[:, c] if len(probs) else np.nan

    out_csv = Path(args.results_dir) / "predictions_test.csv"
    out_df.to_csv(out_csv, index=False)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()