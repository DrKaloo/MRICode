
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
                                  apply_orientation, inv_ornt_aff)

from resnet3d_OASIS import ModelConfig, build_model
from medicalnet3d import MedicalNetConfig, build_medicalnet
from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2

AX_NAME = {"L": "left-right", "R": "left-right", "A": "ant-post", "P": "ant-post",
           "S": "sup-inf", "I": "sup-inf"}
PLANE_OF = {"left-right": "sagittal", "ant-post": "coronal", "sup-inf": "axial"}


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
    name, mod = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            name, mod = n, m
    return name, mod


def load_vol(path, reorient=None):
    """Returns the volume, its affine and its axis codes, both transformed
    together so the mask and the data always live in the same frame."""
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 4:
        vol = np.squeeze(vol)
    aff = img.affine
    if reorient is not None and tuple(nib.aff2axcodes(aff)) != tuple(reorient):
        t = ornt_transform(io_orientation(aff), axcodes2ornt(tuple(reorient)))
        vol = apply_orientation(vol, t)
        aff = aff @ inv_ornt_aff(t, vol.shape)      # keep the affine in step
    return np.nan_to_num(vol, nan=0.0).astype(np.float32), aff, nib.aff2axcodes(aff)


def centre_crop(a, s):
    D, H, W = a.shape
    if s >= min(D, H, W):
        return a
    d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
    return a[d0:d0 + s, h0:h0 + s, w0:w0 + s]


def make_roi(shape, centre_vox, half_vox, mirror_axis=None):
    """Box in voxel space. If mirror_axis is given, a second box is added at the
    mirrored position so the region is bilateral."""
    m = np.zeros(shape, dtype=bool)
    def put(c):
        sl = tuple(slice(max(0, int(c[i] - half_vox[i])),
                         min(shape[i], int(c[i] + half_vox[i]) + 1)) for i in range(3))
        m[sl] = True
    put(centre_vox)
    if mirror_axis is not None:
        c2 = list(centre_vox)
        c2[mirror_axis] = shape[mirror_axis] - 1 - centre_vox[mirror_axis]
        put(c2)
    return m


def plane_labels(axcodes):
    """Which anatomical plane you get by fixing each array axis."""
    return [PLANE_OF[AX_NAME[c]] for c in axcodes]


def atlas_sheet(mean_vol, axcodes, axis, out_png, n=16):
    """Contact sheet through one axis with voxel indices printed, so the
    hippocampus can be located and its index read off directly."""
    labs = plane_labels(axcodes)
    idxs = np.linspace(int(0.20 * mean_vol.shape[axis]),
                       int(0.80 * mean_vol.shape[axis]), n).astype(int)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.1 * rows))
    for ax, i in zip(axes.ravel(), idxs):
        sl = [slice(None)] * 3
        sl[axis] = i
        ax.imshow(np.rot90(mean_vol[tuple(sl)]), cmap="gray")
        ax.set_title(f"axis{axis} = {i}", fontsize=10)
        ax.axis("off")
    for ax in axes.ravel()[len(idxs):]:
        ax.axis("off")
    fig.suptitle(f"OAS1 group mean, {labs[axis]} plane (fixing array axis {axis}, "
                 f"which is {AX_NAME[axcodes[axis]]})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def render_roi(mean_vol, mask, axcodes, title, out_png):
    labs = plane_labels(axcodes)
    c = np.array([int(np.argmax(mask.sum(axis=tuple(x for x in range(3) if x != a)))) if mask.any() else mean_vol.shape[a]//2 for a in range(3)])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for a in range(3):
        sl = [slice(None)] * 3
        sl[a] = c[a]
        axes[a].imshow(np.rot90(mean_vol[tuple(sl)]), cmap="gray")
        axes[a].contour(np.rot90(mask[tuple(sl)]), levels=[0.5],
                        colors="red", linewidths=1.8)
        axes[a].set_title(f"{labs[a]}, axis{a}={c[a]}")
        axes[a].axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_heatmap(brain, cam, mask, axcodes, title, out_png, alpha=0.45):
    """Heatmap overlay in three planes through the CAM's own peak, plus the
    region outline for reference."""
    labs = plane_labels(axcodes)
    c = np.unravel_index(int(np.argmax(cam)), cam.shape)
    b = (brain - brain.min()) / (brain.max() - brain.min() + 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for a in range(3):
        sl = [slice(None)] * 3
        sl[a] = c[a]
        axes[a].imshow(np.rot90(b[tuple(sl)]), cmap="gray")
        im = axes[a].imshow(np.rot90(cam[tuple(sl)]), cmap="jet",
                            alpha=alpha, vmin=0.0, vmax=1.0)
        if mask is not None:
            axes[a].contour(np.rot90(mask[tuple(sl)]), levels=[0.5],
                            colors="white", linewidths=1.4, linestyles="--")
        axes[a].set_title(f"{labs[a]}, axis{a}={c[a]}")
        axes[a].axis("off")
    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.02)
    cb.set_label("Grad-CAM attention (normalised)", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
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
                            align_corners=False).squeeze().cpu().numpy()
        rng = cam.max() - cam.min()
        return (cam - cam.min()) / (rng + 1e-8), t, float(
            torch.softmax(out.detach(), 1)[0, -1].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--outdir", default="analysis_out/gradcam")
    ap.add_argument("--atlas", action="store_true",
                    help="Render labelled slice contact sheets and stop.")
    ap.add_argument("--qc", action="store_true",
                    help="Render the chosen region on the group mean and stop.")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--reorient", default=None)
    ap.add_argument("--roi_centre_vox", nargs=3, type=int, default=None,
                    metavar=("I", "J", "K"))
    ap.add_argument("--roi_half_vox", nargs=3, type=int, default=[12, 14, 10])
    ap.add_argument("--mirror_axis", type=int, default=None,
                    help="Array axis that is left-right, to make the region "
                         "bilateral. Printed for you by --atlas.")
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--n_examples", type=int, default=4,
                    help="Individual subject heatmaps to save per category.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    reo = tuple(args.reorient.upper()) if args.reorient else None

    df = pd.read_csv(Path(args.splits_dir) / "test.csv")
    df = df[df["subject_id"].astype(str).str.startswith("OAS1")].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("No OAS1 subjects. Group CAM is OAS1 only; see the docstring.")

    v0, aff0, ax0 = load_vol(df["scan_path"].iloc[0], reo)
    labs = plane_labels(ax0)
    print(f"{len(df)} OAS1 subjects | shape {v0.shape} | axcodes {ax0}")
    for a in range(3):
        print(f"  array axis {a} is {AX_NAME[ax0[a]]}; fixing it gives a {labs[a]} slice")
    lr_axis = [a for a in range(3) if AX_NAME[ax0[a]] == "left-right"]
    print(f"  left-right axis is array axis {lr_axis[0] if lr_axis else '?'}"
          f"  -> pass --mirror_axis {lr_axis[0] if lr_axis else 0} for a bilateral region\n")

    # group mean, needed by atlas, qc and the group CAM figures
    acc, n, boxes = np.zeros_like(v0, dtype=np.float64), 0, set()
    for p in df["scan_path"]:
        v, _, _ = load_vol(p, reo)
        if v.shape != v0.shape:
            continue
        acc += v; n += 1
        boxes.add(tuple(round(float(z), 3)
                        for z in nib.load(p).header.get_zooms()[:3]))
    mean_vol = acc / max(1, n)
    print(f"group mean over {n} subjects, {len(boxes)} distinct voxel sizes")
    if len(boxes) > 3:
        print("  WARNING: these subjects are not in a shared space; group CAM invalid.")

    if args.atlas:
        for a in range(3):
            png = outdir / f"atlas_axis{a}_{labs[a]}.png"
            atlas_sheet(mean_vol, ax0, a, png)
            print(f"  wrote {png}")
        print("\nFind the hippocampus on the CORONAL sheet: medial temporal lobe,")
        print("below and lateral to the thalamus, just above the parahippocampal")
        print("gyrus. Note the axis index of the slice where it is clearest, then")
        print("read the other two indices off the sagittal and axial sheets.")
        print("Pass all three as --roi_centre_vox I J K and rerun with --qc.") # extra
        return

    if args.roi_centre_vox is None:
        raise SystemExit("Pass --roi_centre_vox I J K. Run --atlas first to find them.")
    mask = make_roi(v0.shape, args.roi_centre_vox, args.roi_half_vox, args.mirror_axis)
    print(f"region: {int(mask.sum())} voxels ({100*mask.mean():.2f} per cent of volume)"
          f"{' bilateral' if args.mirror_axis is not None else ' unilateral'}")

    if args.qc:
        png = outdir / "roi_qc.png"
        render_roi(mean_vol, mask, ax0,
                   f"OAS1 group mean, n={n}. Region from visual localisation.", png)
        print(f"\nWrote {png}. Confirm the outline is on the hippocampus before running the CAM.")
        return

    if not args.ckpts:
        raise SystemExit("Pass --ckpts, or use --atlas / --qc first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_c = centre_crop(mask, args.patch)
    mean_c = centre_crop(mean_vol, args.patch)
    rows = []
    cam_sum = {k: np.zeros_like(mean_c, dtype=np.float64) for k in
               ("all", "correct", "incorrect", "CN", "AD", "under70", "over80")}
    cam_n = {k: 0 for k in cam_sum}
    saved = {}

    for cp in args.ckpts:
        try:
            ck = torch.load(cp, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(cp, map_location="cpu")
        cfg_d = ck.get("cfg", {}) or {}
        k = int(ck.get("num_classes", cfg_d.get("out_dim", 2)))
        model = build_from_cfg(cfg_d, k, device).to(device)
        model.load_state_dict(ck["model_state"], strict=True); model.eval(); del ck
        lname, layer = last_conv(model)
        print(f"{Path(cp).parent.name}: CAM target {lname}")
        engine = GradCAM3D(model, layer)

        for _, r in df.iterrows():
            v, _, _ = load_vol(r["scan_path"], reo)
            vc = centre_crop(v, args.patch)
            x = torch.from_numpy(vc).unsqueeze(0).unsqueeze(0).to(device)
            cam, pred, p_ad = engine(x)
            lab = int(r["label"]); ok = int(pred == lab); age = float(r.get("age", np.nan))
            rows.append({"fold": Path(cp).parent.name, "subject_id": r["subject_id"],
                         "age": age, "label": lab, "pred": pred, "correct": ok,
                         "p_ad": round(p_ad, 4), "cam_mean": float(cam.mean()),
                         "roi_mean": float(cam[mask_c].mean()),
                         "roi_ratio": float(cam[mask_c].mean() / (cam.mean() + 1e-8))})
            for key, cond in (("all", True), ("correct", ok == 1), ("incorrect", ok == 0),
                              ("CN", lab == 0), ("AD", lab == k - 1),
                              ("under70", age < 70), ("over80", age >= 80)):
                if cond:
                    cam_sum[key] += cam; cam_n[key] += 1
            for tag, cond in (("true_AD_correct", lab == k - 1 and ok),
                              ("true_CN_correct", lab == 0 and ok),
                              ("misclassified", not ok)):
                if cond and len(saved.get(tag, [])) < args.n_examples:
                    saved.setdefault(tag, []).append(
                        (r["subject_id"], age, lab, pred, p_ad, vc.copy(), cam.copy()))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "gradcam_stats.csv", index=False)

    figdir = outdir / "figures"; figdir.mkdir(exist_ok=True)
    for key in cam_sum:
        if cam_n[key] < 5:
            continue
        g = cam_sum[key] / cam_n[key]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        render_heatmap(mean_c, g, mask_c, np.array(ax0),
                       f"Group-mean Grad-CAM: {key} (n={cam_n[key]} subject-folds)",
                       figdir / f"groupmean_{key}.png")
        np.save(figdir / f"groupmean_{key}.npy", g)
    for tag, items in saved.items():
        for sid, age, lab, pred, p_ad, vc, cam in items:
            render_heatmap(vc, cam, mask_c, np.array(ax0),
                           f"{sid} | age {age:.0f} | true {lab} pred {pred} "
                           f"| p(AD)={p_ad:.2f}", figdir / f"{tag}_{sid}.png")


    print("roi_ratio 1.0 = the region gets exactly its share of attention by volume.\n")
    print(out.groupby(["label", "correct"]).agg(
        n=("roi_ratio", "size"), roi_ratio=("roi_ratio", "mean"),
        roi_sd=("roi_ratio", "std")).round(3).to_string())
    y, o = out[out.age < 70], out[out.age >= 80]
    if len(y) >= 10 and len(o) >= 10:
        print(f"\nunder 70 (n={len(y)}) {y.roi_ratio.mean():.3f}   "
              f"over 80 (n={len(o)}) {o.roi_ratio.mean():.3f}   "
              f"difference {o.roi_ratio.mean()-y.roi_ratio.mean():+.3f}")
    print(f"\nWrote {outdir/'gradcam_stats.csv'} and {figdir} "
          f"({len([f for f in figdir.glob('*.png')])} figures)")


if __name__ == "__main__":
    main()

