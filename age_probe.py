
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from dataset import BrainMRIDataset
from resnet3d_OASIS import ModelConfig, build_model
from medicalnet3d import MedicalNetConfig, build_medicalnet
from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2


class CenterCrop3D:
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, x):
        _, D, H, W = x.shape
        s = self.size
        if s >= D and s >= H and s >= W:
            return x
        d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
        return x[:, d0:d0 + s, h0:h0 + s, w0:w0 + s]


def collate(batch):
    return {"x": torch.stack([b["x"] for b in batch]),
            "y": [b["y"] for b in batch],
            "meta": [b["meta"] for b in batch]}


def load_ckpt(p):
    try:
        return torch.load(str(p), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(p), map_location="cpu")


def build_from_cfg(cfg_d, k, device):
    if "model_depth" in cfg_d:
        return build_medicalnet(MedicalNetConfig(
            model_depth=int(cfg_d.get("model_depth", 18)),
            shortcut_type=str(cfg_d.get("shortcut_type", "A")),
            pretrained_path=None, in_channels=int(cfg_d.get("in_channels", 1)),
            out_dim=k, dropout=float(cfg_d.get("dropout", 0.2)),
            no_cuda=(device.type != "cuda")))
    if "width_mult" in cfg_d:
        return build_mobilenet3d_v2(MobileNet3DV2Config(
            width_mult=float(cfg_d.get("width_mult", 1.0)),
            pretrained_path=None, in_channels=int(cfg_d.get("in_channels", 1)),
            out_dim=k, dropout=float(cfg_d.get("dropout", 0.2))))
    return build_model(ModelConfig(
        backbone=str(cfg_d.get("backbone", "r3d_18")), pretrained=False,
        in_channels=int(cfg_d.get("in_channels", 1)), out_dim=k,
        dropout=float(cfg_d.get("dropout", 0.2))))


def relocate(stored, roots):
    """Checkpoints record the PyCharm-tree splits path, which no longer exists.
    Rebuild it by matching the longest trailing run of path segments against
    each supplied root."""
    if not stored:
        return None
    p = Path(str(stored).replace("\\", "/"))
    if p.is_dir():
        return p
    parts = p.parts
    for root in roots:
        r = Path(root)
        for i in range(len(parts) - 1, 0, -1):
            cand = r.joinpath(*parts[i:])
            if cand.is_dir():
                return cand
    return None


@torch.no_grad()
def embed(model, csv_path, patch, device, batch_size):
    """Return embeddings (the input to `fc`), labels, ages, subject ids."""
    import inspect as _i; _p = _i.signature(BrainMRIDataset.__init__).parameters; _kw = {"transform": CenterCrop3D(patch), "return_dict": True, "cache_size": 0}; ds = BrainMRIDataset(str(csv_path), **{k: v for k, v in _kw.items() if k in _p})
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=0, collate_fn=collate)
    feats = []

    def pre_hook(module, inputs):
        feats.append(inputs[0].detach().float().cpu())

    h = model.fc.register_forward_pre_hook(pre_hook)
    ys, ages, sids = [], [], []
    for b in dl:
        model(b["x"].to(device))
        ys.append(np.asarray(b["y"], dtype=int))
        ages.append(np.array([np.nan if m["age"] is None else float(m["age"])
                              for m in b["meta"]], dtype=float))
        sids.append(np.array([str(m["patient_id"]) for m in b["meta"]], dtype=object))
    h.remove()

    E = torch.cat(feats, 0).numpy()
    y = np.concatenate(ys)
    a = np.concatenate(ages)
    s = np.concatenate(sids)
    keep = ~np.isnan(a)
    return E[keep], y[keep], a[keep], s[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="analysis_out/run_index.csv")
    ap.add_argument("--splits_roots", nargs="+", required=True)
    ap.add_argument("--out", default="analysis_out/age_probe.csv")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--include_cross", action="store_true",
                    help="Also probe cross-cohort runs. Off by default: H2 is a "
                         "within-cohort claim and this doubles the runtime.")
    ap.add_argument("--probe_train_n", type=int, default=0); ap.add_argument("--smoke", action="store_true",
                    help="Run one fold only, to validate paths before committing.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    idx = pd.read_csv(args.index)
    idx = idx[idx["arm"].isin(["2class", "3class"])]
    if not args.include_cross:
        idx = idx[idx["direction"] == "within"]
    idx = idx.sort_values(["arm", "arch", "train_cohort", "fold"])
    if args.smoke:
        idx = idx.head(1)
    print(f"{len(idx)} runs to probe\n")

    rows = []
    for _, r in idx.iterrows():
        d = Path(r["pred_file"]).parent
        summ = d / "evaluation_summary.json"
        if not summ.is_file():
            print(f"  [skip] no evaluation_summary.json in {d}")
            continue
        info = json.load(open(summ, "r", encoding="utf-8"))
        ck_path = info.get("ckpt")
        if not ck_path or not Path(ck_path).is_file():
            print(f"  [skip] checkpoint missing: {ck_path}")
            continue

        ck = load_ckpt(ck_path)
        cfg_d = ck.get("cfg", {}) or {}
        a = ck.get("args", {}) or {}
        k = int(ck.get("num_classes", cfg_d.get("out_dim", 3)))
        patch = int(a.get("patch_size", 96))

        sp = relocate(ck.get("splits_dir") or a.get("splits_dir"), args.splits_roots)
        if sp is None:
            print(f"  [skip] could not relocate splits for {ck_path}")
            continue
        train_csv, test_csv = sp / "train.csv", sp / "test.csv"
        if not (train_csv.is_file() and test_csv.is_file()):
            print(f"  [skip] train/test missing under {sp}")
            continue

        model = build_from_cfg(cfg_d, k, device).to(device)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        del ck

        Etr, ytr, atr, _ = embed(model, train_csv, patch, device, args.batch_size)
        Ete, yte, ate, _ = embed(model, test_csv, patch, device, args.batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.probe_train_n and len(atr) > args.probe_train_n:
            _s = np.random.default_rng(1337).choice(len(atr), args.probe_train_n, replace=False)
            Etr, ytr, atr = Etr[_s], ytr[_s], atr[_s]
        sc = StandardScaler().fit(Etr)
        Xtr, Xte = sc.transform(Etr), sc.transform(Ete)

        # age decodability
        ridge = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(Xtr, atr)
        pred_age = ridge.predict(Xte)
        r2 = float(r2_score(ate, pred_age))
        mae = float(mean_absolute_error(ate, pred_age))

        # trivial baseline: always predict the training mean age
        mae_base = float(mean_absolute_error(ate, np.full_like(ate, atr.mean())))

        # disease decodability from the SAME embedding, for comparison
        dis_auc = float("nan")
        try:
            if len(np.unique(ytr)) > 1 and len(np.unique(yte)) > 1:
                lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
                P = lr.predict_proba(Xte)
                if k == 2:
                    dis_auc = float(roc_auc_score(yte, P[:, 1]))
                else:
                    dis_auc = float(roc_auc_score(np.eye(k, dtype=int)[yte], P,
                                                  average="macro", multi_class="ovr"))
        except Exception as e:
            print(f"    [warn] disease probe failed: {e}")

        rows.append({
            "arm": r["arm"], "arch": r["arch"], "cohort": r["train_cohort"],
            "fold": r["fold"], "dim": int(Etr.shape[1]),
            "n_train": int(len(atr)), "n_test": int(len(ate)),
            "age_r2": round(r2, 4), "age_mae": round(mae, 3),
            "age_mae_baseline": round(mae_base, 3),
            "mae_improvement": round(mae_base - mae, 3),
            "disease_auc_from_embedding": round(dis_auc, 4) if dis_auc == dis_auc else None,
            "alpha": float(ridge.alpha_),
        })
        print(f"  {r['arm']:<7} {r['arch']:<20} {r['train_cohort']:<6} fold {r['fold']}: "
              f"age R2 {r2:+.3f}  MAE {mae:.2f}y (baseline {mae_base:.2f})  "
              f"disease AUC {dis_auc:.3f}")

    if not rows:
        raise SystemExit("Nothing probed. Check --splits_roots.")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 96)
    print("4.7  AGE DECODABILITY FROM THE LEARNED EMBEDDING")
    print("=" * 96)
    agg = out.groupby(["arm", "arch", "cohort"]).agg(
        n=("age_r2", "size"),
        age_r2=("age_r2", "mean"), age_r2_sd=("age_r2", "std"),
        age_mae=("age_mae", "mean"), mae_gain=("mae_improvement", "mean"),
        disease_auc=("disease_auc_from_embedding", "mean")).round(3)
    print(agg.to_string())

    print("\nUnderstanding this:")
    print("  age_r2 near 0 or negative  the embedding carries no usable age signal.")
    print("  mae_gain                    years of error saved against always")
    print("                              predicting the training mean age. This is")
    print("                              the honest effect size; R2 alone is easy to")
    print("                              over-read on small test folds.")
    print("  disease_auc vs age_r2       the H2 comparison. Age decoded well while")
    print("                              diagnosis decodes weakly, from the ==/Same")
    print("                              vector, is direct evidence that the")
    print("                              representation encodes age more strongly")
    print("                              than pathology.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()


