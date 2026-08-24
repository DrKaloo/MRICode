

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.orientations import (io_orientation, axcodes2ornt, ornt_transform,
                                  apply_orientation)
from scipy.stats import rankdata

from resnet3d_OASIS import ModelConfig, build_model
from medicalnet3d import MedicalNetConfig, build_medicalnet
from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2


def fast_auc(y, s):
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


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


def centre_crop(x, size):
    _, D, H, W = x.shape
    s = int(size)
    if s >= D and s >= H and s >= W:
        return x
    d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
    return x[:, d0:d0 + s, h0:h0 + s, w0:w0 + s]


def load_volume(path, patch, target_axcodes=None):
    """Replicates dataset.py's chain, with an optional reorientation inserted
    before the permute, which is where the axis order is located."""
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = np.squeeze(vol)
    src = nib.aff2axcodes(img.affine)
    if target_axcodes is not None and tuple(src) != tuple(target_axcodes):
        t = ornt_transform(io_orientation(img.affine),
                           axcodes2ornt(tuple(target_axcodes)))
        vol = apply_orientation(vol, t)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(vol)).permute(2, 0, 1).unsqueeze(0)
    return centre_crop(x, patch), src


@torch.no_grad()
def run(model, paths, patch, device, target, batch_size=2):
    probs, srcs, buf = [], [], []
    for i, p in enumerate(paths):
        x, src = load_volume(p, patch, target)
        buf.append(x)
        srcs.append(src)
        if len(buf) == batch_size or i == len(paths) - 1:
            out = model(torch.stack(buf).to(device))
            probs.append(torch.softmax(out, dim=1).float().cpu().numpy())
            buf = []
    return np.concatenate(probs), srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--splits_dir", required=True,
                    help="Directory containing the OASIS test.csv to evaluate on.")
    ap.add_argument("--target_axcodes", default="IPL",
                    help="Axis order to align to. IPL is ADNI's, measured across "
                         "all 458 scans.")
    ap.add_argument("--out", default="analysis_out/reorient_test.csv")
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    target = tuple(args.target_axcodes.upper())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_csv = Path(args.splits_dir) / "test.csv"
    df = pd.read_csv(test_csv)
    paths = df["scan_path"].astype(str).tolist()
    sid = df["subject_id"].astype(str)
    is_oas1 = sid.str.startswith("OAS1").to_numpy()
    print(f"{len(df)} subjects, {int(is_oas1.sum())} OAS1 and "
          f"{int((~is_oas1).sum())} OAS2, target axis order {target}\n")

    rows = []
    for cp in args.ckpts:
        try:
            ck = torch.load(cp, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(cp, map_location="cpu")
        cfg_d = ck.get("cfg", {}) or {}
        a = ck.get("args", {}) or {}
        k = int(ck.get("num_classes", cfg_d.get("out_dim", 3)))
        patch = int(a.get("patch_size", 96))
        ad = k - 1
        y = (df["label"].to_numpy() == ad).astype(int)

        model = build_from_cfg(cfg_d, k, device).to(device)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval()
        del ck

        p_base, srcs = run(model, paths, patch, device, None, args.batch_size)
        p_re, _ = run(model, paths, patch, device, target, args.batch_size)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        def three(p):
            return (fast_auc(y, p[:, ad]),
                    fast_auc(y[is_oas1], p[is_oas1, ad]),
                    fast_auc(y[~is_oas1], p[~is_oas1, ad]))

        b_all, b_1, b_2 = three(p_base)
        r_all, r_1, r_2 = three(p_re)
        rows.append({
            "ckpt": cp,
            "auc_all_baseline": round(b_all, 4), "auc_all_reoriented": round(r_all, 4),
            "delta_all": round(r_all - b_all, 4),
            "auc_OAS1_baseline": round(b_1, 4), "auc_OAS1_reoriented": round(r_1, 4),
            "delta_OAS1": round(r_1 - b_1, 4),
            "auc_OAS2_baseline": round(b_2, 4), "auc_OAS2_reoriented": round(r_2, 4),
            "delta_OAS2": round(r_2 - b_2, 4),
            "n_axcodes_seen": len(set(srcs)),
        })
        print(f"  {Path(cp).parent.name}: all {b_all:.3f} -> {r_all:.3f} "
              f"({r_all - b_all:+.3f}) | OAS1 {b_1:.3f} -> {r_1:.3f} "
              f"({r_1 - b_1:+.3f}) | OAS2 {b_2:.3f} -> {r_2:.3f} ({r_2 - b_2:+.3f})")

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n" + "=" * 88)
    print("Reorentation Test")
    print("=" * 88)
    for col in ("all", "OAS1", "OAS2"):
        b = out[f"auc_{col}_baseline"].mean()
        r = out[f"auc_{col}_reoriented"].mean()
        d = out[f"delta_{col}"]
        print(f"  {col:<5} baseline {b:.4f}  reoriented {r:.4f}  "
              f"delta {d.mean():+.4f} (sd {d.std(ddof=1):.4f}, "
              f"{int((d > 0).sum())} of {len(d)} folds improved)")

    dm = out["delta_OAS1"].mean()
    print()
    if dm > 0.05:
        print("SUBSTANTIAL RECOVERY. Axis-order mismatch accounted for a measurable")
        print("share of the apparent cross-cohort failure, and it is correctable at")
        print("test time with no retraining. This is a headline result: report the")
        print("baseline as the finding and the corrected figure as the decomposition,")
        print("exactly as you do with the oracle threshold.")
    elif dm > 0.02:
        print("PARTIAL RECOVERY. Real but modest. Report it as a quantified component")
        print("of the cross-cohort gap alongside the bounding-box mismatch, which this")
        print("does not address.")
    else:
        print("NO RECOVERY. Orientation is a genuine pipeline defect but is not what")
        print("drives the cross-cohort collapse. That is worth knowing and worth")
        print("reporting: it closes the question and points at the bounding-box and")
        print("vendor-preprocessing differences instead.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
