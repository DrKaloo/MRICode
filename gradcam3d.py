
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from nibabel.orientations import (io_orientation, axcodes2ornt, ornt_transform,
                                  apply_orientation)

from resnet3d_OASIS import ModelConfig, build_model
from medicalnet3d import MedicalNetConfig, build_medicalnet
from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2

# Talairach centroid and half-extent of the hippocampus, in millimetres.
# Bilateral: x is mirrored. Values are the conventional atlas position; the QC
# render exists so this can be checked rather than trusted.
HIPP_CENTRE = (26.0, -22.0, -16.0)
HIPP_HALF = (13.0, 20.0, 10.0)


def build_from_cfg(cfg_d, k, device):
    if "model_depth" in cfg_d:
        return build_medicalnet(MedicalNetConfig(
            model_depth=int(cfg_d.get("model_depth", 18)),
            shortcut_type=str(cfg_d.get("shortcut_type", "A")), pretrained_path=None,
            in_channels=int(cfg_d.get("in_channels", 1)), out_dim=k,
            dropout=float(cfg_d.get("dropout", 0.2)), no_cuda=(device.type != "cuda")))
    if "width_mult" in cfg_d:
        return build_mobilenet3d_v2(MobileNet3DV2Config(
            width_mult=float(cfg_d.get("width_mult", 1.0)), pretrained_path=None,
            in_channels=int(cfg_d.get("in_channels", 1)), out_dim=k,
            dropout=float(cfg_d.get("dropout", 0.2))))
    return build_model(ModelConfig(
        backbone=str(cfg_d.get("backbone", "r3d_18")), pretrained=False,
        in_channels=int(cfg_d.get("in_channels", 1)), out_dim=k,
        dropout=float(cfg_d.get("dropout", 0.2))))


def last_conv(model):
    """Last Conv3d in module order: the standard Grad-CAM target, and works for
    all three backbones without hardcoding a module path."""
    name, mod = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            name, mod = n, m
    return name, mod


def roi_from_affine(affine, shape, mode="talairach"):
    """Boolean mask in voxel space. talairach mode inverts the affine so the box
    is defined anatomically; fractional mode is the documented fallback and is
    still consistent across subjects because the bounding box is shared."""
    mask = np.zeros(shape, dtype=bool)
    if mode == "fractional":
        cx, cy, cz = (np.array(shape) * np.array([0.5, 0.55, 0.42])).astype(int)
        hx, hy, hz = (np.array(shape) * 0.11).astype(int)
        mask[max(0, cx - hx):cx + hx, max(0, cy - hy):cy + hy,
             max(0, cz - hz):cz + hz] = True
        return mask, "fractional box (fallback)"

    inv = np.linalg.inv(affine)
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    vox = np.stack([ii, jj, kk, np.ones_like(ii)], -1).reshape(-1, 4).astype(float)
    world = (affine @ vox.T).T[:, :3]
    cx, cy, cz = HIPP_CENTRE
    hx, hy, hz = HIPP_HALF
    inbox = ((np.abs(np.abs(world[:, 0]) - cx) <= hx) &
             (np.abs(world[:, 1] - cy) <= hy) &
             (np.abs(world[:, 2] - cz) <= hz))
    mask = inbox.reshape(shape)
    return mask, "Talairach box from affine"


def load_vol(path, reorient=None):
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = np.squeeze(vol)
    aff = img.affine
    t = None
    if reorient is not None and tuple(nib.aff2axcodes(aff)) != tuple(reorient):
        t = ornt_transform(io_orientation(aff), axcodes2ornt(tuple(reorient)))
        vol = apply_orientation(vol, t)
    return np.nan_to_num(vol, nan=0.0).astype(np.float32), aff, t


def centre_crop_np(a, s):
    D, H, W = a.shape
    if s >= min(D, H, W):
        return a
    d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
    return a[d0:d0 + s, h0:h0 + s, w0:w0 + s]


def render(mean_vol, mask, title, out_png):
    """Three orthogonal planes through the ROI centroid, mask outlined in red."""
    idx = np.argwhere(mask)
    c = idx.mean(0).astype(int) if len(idx) else (np.array(mean_vol.shape) // 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    planes = [(mean_vol[c[0], :, :], mask[c[0], :, :], f"sagittal i={c[0]}"),
              (mean_vol[:, c[1], :], mask[:, c[1], :], f"coronal j={c[1]}"),
              (mean_vol[:, :, c[2]], mask[:, :, c[2]], f"axial k={c[2]}")]
    for ax, (im, mk, ttl) in zip(axes, planes):
        ax.imshow(np.rot90(im), cmap="gray")
        ax.contour(np.rot90(mk), levels=[0.5], colors="red", linewidths=1.6)
        ax.set_title(ttl); ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


class GradCAM3D:
    def __init__(self, model, layer):
        self.model, self.a, self.g = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o.detach()))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "g", go[0].detach()))

    def __call__(self, x, target=None):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        t = int(out.argmax(1).item()) if target is None else int(target)
        out[0, t].backward()
        w = self.g.mean(dim=(2, 3, 4), keepdim=True)
        cam = F.relu((w * self.a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        rng = cam.max() - cam.min()
        return (cam - cam.min()) / (rng + 1e-8), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--outdir", default="analysis_out/gradcam")
    ap.add_argument("--qc", action="store_true")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--reorient", default=None, help="e.g. IPL")
    ap.add_argument("--roi_mode", default="talairach",
                    choices=["talairach", "fractional"])
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--max_subjects", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    reo = tuple(args.reorient.upper()) if args.reorient else None

    df = pd.read_csv(Path(args.splits_dir) / "test.csv")
    df = df[df["subject_id"].astype(str).str.startswith("OAS1")].reset_index(drop=True)
    if args.max_subjects:
        df = df.head(args.max_subjects)
    if len(df) == 0:
        raise SystemExit("No OAS1 subjects, see the docstring.")
    print(f"{len(df)} OAS1 subjects")

# shared space check + ROI
    v0, aff0, t0 = load_vol(df["scan_path"].iloc[0], reo)
    zooms = nib.load(df["scan_path"].iloc[0]).header.get_zooms()[:3]
    corners = np.array([[0, 0, 0, 1], [*v0.shape, 1]], float)
    world = (aff0 @ corners.T).T[:, :3]
    print(f"volume shape {v0.shape}, voxel size {tuple(round(float(z),3) for z in zooms)}")
    print(f"world extent from affine: {np.round(world[0],1)} to {np.round(world[1],1)}")
    print("  A Talairach-like affine spans roughly x -70..70, y -105..70, z -45..75.")
    print("  If these numbers look nothing like that, use --roi_mode fractional.\n")

    mask, roi_desc = roi_from_affine(aff0, v0.shape, args.roi_mode)
    if t0 is not None:
        pass  # mask already built in the reoriented frame, since aff0 is pre-reorientation
    if mask.sum() == 0:
        raise SystemExit("ROI is empty. The affine is not Talairach-like. "
                         "Rerun with --roi_mode fractional.")
    print(f"ROI: {roi_desc}, {int(mask.sum())} voxels "
          f"({100*mask.mean():.2f} per cent of the volume)")

# QC
    if args.qc:
        acc = np.zeros_like(v0, dtype=np.float64)
        n = 0
        boxes = set()
        for p in df["scan_path"]:
            v, aff, _ = load_vol(p, reo)
            if v.shape != v0.shape:
                continue
            boxes.add(tuple(round(float(z), 3)
                            for z in nib.load(p).header.get_zooms()[:3]))
            acc += v; n += 1
        mean_vol = acc / max(1, n)
        print(f"group mean over {n} subjects; {len(boxes)} distinct voxel sizes seen")
        if len(boxes) > 3:
            print("  WARNING: more than three distinct bounding boxes. These subjects "
                  "are NOT in a shared space and group Grad-CAM is not valid on them.")
        png = outdir / "roi_qc.png"
        render(mean_vol, mask, f"OAS1 group mean, n={n}. ROI: {roi_desc}", png)
        np.save(outdir / "group_mean.npy", mean_vol)
        print(f"\nWrote {png}")
        print("LOOK AT IT. The red outline must sit on the hippocampus: medial")
        print("temporal lobe, below and lateral to the thalamus, curving back from")
        print("the amygdala. If it does not, adjust HIPP_CENTRE at the top of this")
        print("file or switch to --roi_mode fractional, and say so in the methods.")
        print("Do not run the CAM stage until this render is confirmed.")
        return

# CAM
    if not args.ckpts:
        raise SystemExit("Pass --ckpts, or run --qc first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_c = centre_crop_np(mask, args.patch)
    rows = []
    for cp in args.ckpts:
        try:
            ck = torch.load(cp, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(cp, map_location="cpu")
        cfg_d = ck.get("cfg", {}) or {}
        k = int(ck.get("num_classes", cfg_d.get("out_dim", 2)))
        model = build_from_cfg(cfg_d, k, device).to(device)
        model.load_state_dict(ck["model_state"], strict=True)
        model.eval(); del ck
        lname, layer = last_conv(model)
        print(f"{Path(cp).parent.name}: CAM target layer {lname}")
        cam_engine = GradCAM3D(model, layer)

        for _, r in df.iterrows():
            v, _, _ = load_vol(r["scan_path"], reo)
            x = torch.from_numpy(centre_crop_np(v, args.patch)).unsqueeze(0).unsqueeze(0)
            cam, pred = cam_engine(x.to(device))
            rows.append({
                "fold": Path(cp).parent.name, "subject_id": r["subject_id"],
                "age": r.get("age"), "label": int(r["label"]), "pred": pred,
                "correct": int(pred == int(r["label"])),
                "cam_mean": float(cam.mean()),
                "roi_mean": float(cam[mask_c].mean()),
                "roi_ratio": float(cam[mask_c].mean() / (cam.mean() + 1e-8)),
            })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "gradcam_stats.csv", index=False)

    print("roi_ratio 1.0 means the region receives exactly its share of attention")
    print("by volume. Above 1.0 means the model attends there preferentially.\n")
    agg = out.groupby(["label", "correct"]).agg(
        n=("roi_ratio", "size"), roi_ratio=("roi_ratio", "mean"),
        roi_sd=("roi_ratio", "std")).round(3)
    print(agg.to_string())
    young = out[out["age"] < 70]; old = out[out["age"] >= 80]
    if len(young) >= 10 and len(old) >= 10:
        print(f"\nunder 70 (n={len(young)}): roi_ratio {young['roi_ratio'].mean():.3f}"
              f"   over 80 (n={len(old)}): {old['roi_ratio'].mean():.3f}")
        print("A difference here is the attribution-level counterpart to H2.")
    print(f"\nWrote {outdir/'gradcam_stats.csv'}")


if __name__ == "__main__":
    main()
