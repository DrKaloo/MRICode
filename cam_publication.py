"""
cam_publication.py

Draelos and Carin (arXiv:2011.08891) proved that Grad-CAM's gradient-averaging
step can highlight locations the model did not use, and showed in volumetric
medical imaging (AxialNet, chest CT, arXiv:2111.12215) that this "creates the
incorrect impression that the model has focused on the wrong organ". Given that
the February hippocampal analysis in this project failed precisely by claiming a
wrong location, the faithful method is the appropriate default.

  HiResCAM   cam = ReLU( sum_c  A_c * dY/dA_c )      <---- default
             Hadamard product, no gradient averaging. Provably highlights only
             locations that contributed to the class score for a CNN ending in a
             single fully connected layer, which is exactly the head used by all
             three backbones here (dropout + one Linear).

  Grad-CAM   cam = ReLU( sum_c  mean(dY/dA_c) * A_c )
             Included for comparison because it is what the field reports.
             Smoother and larger; the difference between the two is itself worth
             a figure.

  Grad-CAM++ second-order weighting, better when a class appears in several
             places. Included as a third view.

Sanity Check (Adebayo et al., NeurIPS 2018). A saliency map that barely changes
when the model's weights are randomised is not explaining the model. Only about
28 per cent of medical imaging XAI studies report any faithfulness evaluation
(systematic review, arXiv:2601.00990), so including one is both cheap and rare.
--sanity reruns the CAM with the target block's weights reinitialised and reports
the correlation with the real map. Correlation near zero is a pass; correlation
near one means the map reflects image structure rather than the model.

Rendering. Built for print rather than for a screenshot:
  - multi-slice montage rather than three arbitrary planes
  - perceptually uniform colormap, not jet, which manufactures false edges
  - alpha ramps with attention, so anatomy stays visible and low-attention
    regions are transparent instead of washed blue
  - values below a percentile are fully transparent, so the figure shows where
    the model looked rather than a uniform colour field
  - anatomical side labels derived from the affine, never assumed
  - shared colour scale with a labelled colourbar

Usage:
    python cam_publication.py --ckpts ...\\best.pt ^
      --splits_dir "...\\res_128_cnad\\all\\validated" --reorient IPL ^
      --roi_centre_vox 76 70 88 --mirror_axis 2 ^
      --method hirescam --outdir analysis_out\\cam --sanity
"""

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
from matplotlib.colors import LinearSegmentedColormap
from nibabel.orientations import (io_orientation, axcodes2ornt, ornt_transform,
                                  apply_orientation, inv_ornt_aff)

from resnet3d_OASIS import ModelConfig, build_model
from medicalnet3d import MedicalNetConfig, build_medicalnet
from mobilenet3d_v2 import MobileNet3DV2Config, build_mobilenet3d_v2

AX_NAME = {"L": "left-right", "R": "left-right", "A": "ant-post",
           "P": "ant-post", "S": "sup-inf", "I": "sup-inf"}
PLANE = {"left-right": "sagittal", "ant-post": "coronal", "sup-inf": "axial"}
# ends of each axis, for anatomical labelling: axcode letter is where the axis POINTS
END = {"L": ("R", "L"), "R": ("L", "R"), "A": ("P", "A"),
       "P": ("A", "P"), "S": ("I", "S"), "I": ("S", "I")}


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
    n_, m_ = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            n_, m_ = n, m
    return n_, m_


class VolumetricCAM:
    """HiResCAM, Grad-CAM and Grad-CAM++ from one forward and one backward pass."""

    def __init__(self, model, layer):
        self.model = model
        self.a = self.g = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "a", o))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "g", go[0].detach()))

    def __call__(self, x, method="hirescam", target=None):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        t = int(out.argmax(1).item()) if target is None else int(target)
        out[0, t].backward()
        A = self.a.detach()          # [1, C, D, H, W]
        G = self.g                   # [1, C, D, H, W]

        if method == "hirescam":
            # Hadamard product then channel sum: no averaging, faithful
            cam = (A * G).sum(1, keepdim=True)
        elif method == "gradcam":
            w = G.mean(dim=(2, 3, 4), keepdim=True)
            cam = (w * A).sum(1, keepdim=True)
        elif method == "gradcampp":
            g2, g3 = G.pow(2), G.pow(3)
            denom = 2.0 * g2 + (A * g3).sum(dim=(2, 3, 4), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(G)).sum(dim=(2, 3, 4), keepdim=True)
            cam = (w * A).sum(1, keepdim=True)
        else:
            raise ValueError(method)

        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear",
                            align_corners=False).squeeze().detach().cpu().numpy()
        rng = cam.max() - cam.min()
        prob = torch.softmax(out.detach(), 1)[0].cpu().numpy()
        return (cam - cam.min()) / (rng + 1e-8), t, prob


def attention_cmap():
    """Transparent below threshold, then warm and increasingly opaque. Reads as
    'where the model looked' rather than a coloured wash over the whole brain."""
    base = plt.get_cmap("inferno")(np.linspace(0.15, 1.0, 256))
    base[:, -1] = np.clip(np.linspace(-0.15, 1.0, 256), 0, 1) ** 0.8
    return LinearSegmentedColormap.from_list("attention", base)


def montage(brain, cam, mask, axcodes, axis, title, out_png,
            n_slices=9, pct=70, cmap=None):
    """Evenly spaced slices through the brain along one axis, attention overlaid."""
    cmap = cmap or attention_cmap()
    labs = [PLANE[AX_NAME[c]] for c in axcodes]
    b = (brain - np.percentile(brain, 1)) / (np.percentile(brain, 99)
                                             - np.percentile(brain, 1) + 1e-8)
    b = np.clip(b, 0, 1)
    thr = np.percentile(cam, pct)
    shown = np.where(cam >= thr, cam, np.nan)

    lo, hi = END[axcodes[axis]]
    idxs = np.linspace(0.28, 0.72, n_slices) * brain.shape[axis]
    cols = min(n_slices, 5)
    rows = int(np.ceil(n_slices / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.05 * rows),
                             facecolor="black")
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, fi in zip(axes, idxs):
        i = int(fi)
        sl = [slice(None)] * 3
        sl[axis] = i
        ax.imshow(np.rot90(b[tuple(sl)]), cmap="gray", vmin=0, vmax=1,
                  interpolation="bilinear")
        im = ax.imshow(np.rot90(shown[tuple(sl)]), cmap=cmap, vmin=thr, vmax=1.0,
                       interpolation="bilinear")
        if mask is not None:
            ax.contour(np.rot90(mask[tuple(sl)]), levels=[0.5], colors="#00E5FF",
                       linewidths=1.0, linestyles="-")
        ax.set_title(f"{i}", color="white", fontsize=9, pad=3)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off"); ax.set_facecolor("black")
    axes[0].text(0.02, 0.5, lo, color="white", fontsize=11, fontweight="bold",
                 transform=axes[0].transAxes, va="center")
    axes[0].text(0.94, 0.5, hi, color="white", fontsize=11, fontweight="bold",
                 transform=axes[0].transAxes, va="center")
    cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.015)
    cb.set_label(f"attention (top {100-pct}% shown)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    fig.suptitle(f"{title}\n{labs[axis]} montage, cyan outline = region of interest",
                 color="white", fontsize=13, fontweight="bold")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def compare_methods(brain, cams, axcodes, axis, title, out_png, pct=70):
    """One row per attribution method through the same slice. This figure is the
    argument for using HiResCAM: Grad-CAM's map is visibly larger and smoother."""
    cmap = attention_cmap()
    b = np.clip((brain - np.percentile(brain, 1))
                / (np.percentile(brain, 99) - np.percentile(brain, 1) + 1e-8), 0, 1)
    names = list(cams)
    idxs = np.linspace(0.35, 0.65, 4) * brain.shape[axis]
    fig, axes = plt.subplots(len(names), 4, figsize=(11.5, 3.0 * len(names)),
                             facecolor="black", squeeze=False)
    for r, nm in enumerate(names):
        c = cams[nm]
        thr = np.percentile(c, pct)
        shown = np.where(c >= thr, c, np.nan)
        for k, fi in enumerate(idxs):
            i = int(fi)
            sl = [slice(None)] * 3
            sl[axis] = i
            axes[r, k].imshow(np.rot90(b[tuple(sl)]), cmap="gray", vmin=0, vmax=1)
            axes[r, k].imshow(np.rot90(shown[tuple(sl)]), cmap=cmap,
                              vmin=thr, vmax=1.0)
            axes[r, k].axis("off")
            if k == 0:
                axes[r, k].text(-0.06, 0.5, nm, color="white", fontsize=12,
                                fontweight="bold", rotation=90, va="center",
                                transform=axes[r, k].transAxes)
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="black")
    plt.close(fig)


def load_vol(path, reorient=None):
    img = nib.load(path)
    v = img.get_fdata(dtype=np.float32)
    if v.ndim == 4:
        v = np.squeeze(v)
    aff = img.affine
    if reorient and tuple(nib.aff2axcodes(aff)) != tuple(reorient):
        t = ornt_transform(io_orientation(aff), axcodes2ornt(tuple(reorient)))
        v = apply_orientation(v, t)
        aff = aff @ inv_ornt_aff(t, v.shape)
    return np.nan_to_num(v, nan=0.0).astype(np.float32), nib.aff2axcodes(aff)


def crop(a, s):
    D, H, W = a.shape
    if s >= min(D, H, W):
        return a
    d0, h0, w0 = (D - s) // 2, (H - s) // 2, (W - s) // 2
    return a[d0:d0 + s, h0:h0 + s, w0:w0 + s]


def make_roi(shape, c, half, mirror=None):
    m = np.zeros(shape, bool)
    def put(cc):
        m[tuple(slice(max(0, int(cc[i] - half[i])),
                      min(shape[i], int(cc[i] + half[i]) + 1)) for i in range(3))] = True
    put(c)
    if mirror is not None:
        c2 = list(c); c2[mirror] = shape[mirror] - 1 - c[mirror]; put(c2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--outdir", default="analysis_out/cam")
    ap.add_argument("--reorient", default=None)
    ap.add_argument("--method", default="hirescam",
                    choices=["hirescam", "gradcam", "gradcampp"])
    ap.add_argument("--roi_centre_vox", nargs=3, type=int, default=None)
    ap.add_argument("--roi_half_vox", nargs=3, type=int, default=[6, 14, 7])
    ap.add_argument("--mirror_axis", type=int, default=None)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--montage_axis", type=int, default=1)
    ap.add_argument("--pct", type=int, default=70)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"; figs.mkdir(exist_ok=True)
    reo = tuple(args.reorient.upper()) if args.reorient else None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Path(args.splits_dir) / "test.csv")
    df = df[df["subject_id"].astype(str).str.startswith("OAS1")].reset_index(drop=True)
    if not len(df):
        raise SystemExit("No OAS1 subjects. Group attribution requires a shared space.")
    v0, ax0 = load_vol(df["scan_path"].iloc[0], reo)
    print(f"{len(df)} OAS1 subjects | axcodes {ax0} | method {args.method}")

    mask = None
    if args.roi_centre_vox:
        mask = crop(make_roi(v0.shape, args.roi_centre_vox,
                             args.roi_half_vox, args.mirror_axis), args.patch)

    keys = ("all", "correct", "incorrect", "CN", "AD", "under70", "over80")
    acc = {k: np.zeros((args.patch,) * 3) for k in keys}
    cnt = {k: 0 for k in keys}
    brain_acc = np.zeros((args.patch,) * 3)
    rows, examples, sanity = [], [], []

    for cp in args.ckpts:
        try:
            ck = torch.load(cp, map_location="cpu", weights_only=False)
        except TypeError:
            ck = torch.load(cp, map_location="cpu")
        cfg_d = ck.get("cfg", {}) or {}
        k = int(ck.get("num_classes", cfg_d.get("out_dim", 2)))
        model = build_from_cfg(cfg_d, k, dev).to(dev)
        model.load_state_dict(ck["model_state"], strict=True); model.eval(); del ck
        lname, layer = last_conv(model)
        print(f"  {Path(cp).parent.name}: target {lname}")
        engine = VolumetricCAM(model, layer)

        for j, r in df.iterrows():
            v, _ = load_vol(r["scan_path"], reo)
            vc = crop(v, args.patch)
            x = torch.from_numpy(vc).unsqueeze(0).unsqueeze(0).to(dev)
            cam, pred, prob = engine(x, args.method)
            lab = int(r["label"]); ok = int(pred == lab); age = float(r.get("age", np.nan))
            rec = {"fold": Path(cp).parent.name, "subject_id": r["subject_id"],
                   "age": age, "label": lab, "pred": pred, "correct": ok,
                   "p_ad": round(float(prob[-1]), 4), "cam_mean": float(cam.mean())}
            if mask is not None:
                rec["roi_mean"] = float(cam[mask].mean())
                rec["roi_ratio"] = float(cam[mask].mean() / (cam.mean() + 1e-8))
            rows.append(rec)
            for key, cond in (("all", True), ("correct", ok == 1), ("incorrect", ok == 0),
                              ("CN", lab == 0), ("AD", lab == k - 1),
                              ("under70", age < 70), ("over80", age >= 80)):
                if cond:
                    acc[key] += cam; cnt[key] += 1
            brain_acc += vc
            if len(examples) < 3 and lab == k - 1 and ok:
                cams = {m: engine(x, m)[0] for m in ("hirescam", "gradcam", "gradcampp")}
                examples.append((r["subject_id"], age, vc.copy(), cams))

        if args.sanity and not sanity:
            real = engine(torch.from_numpy(crop(load_vol(df["scan_path"].iloc[0], reo)[0],
                                                args.patch)).unsqueeze(0).unsqueeze(0).to(dev),
                          args.method)[0]
            for p in layer.parameters():
                nn.init.normal_(p, 0.0, 0.05)
            rand = engine(torch.from_numpy(crop(load_vol(df["scan_path"].iloc[0], reo)[0],
                                                args.patch)).unsqueeze(0).unsqueeze(0).to(dev),
                          args.method)[0]
            c = float(np.corrcoef(real.ravel(), rand.ravel())[0, 1])
            sanity.append(c)
            print(f"  Sanitycheck: correlation between the real map and the map from "
                  f"randomised weights = {c:.3f}")
            print("    near 0 passes; near 1 means the map reflects image structure, "
                  "not the model.")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(out / f"cam_stats_{args.method}.csv", index=False)
    brain = brain_acc / max(1, len(res))

    for key in keys:
        if cnt[key] < 5:
            continue
        g = acc[key] / cnt[key]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        montage(brain, g, mask, ax0, args.montage_axis,
                f"{args.method.upper()} group mean: {key}  (n = {cnt[key]} subject-folds)",
                figs / f"group_{args.method}_{key}.png", pct=args.pct)
        np.save(figs / f"group_{args.method}_{key}.npy", g)

    for sid, age, vc, cams in examples:
        compare_methods(vc, cams, ax0, args.montage_axis,
                        f"{sid}, age {age:.0f}: attribution method comparison",
                        figs / f"methods_{sid}.png", pct=args.pct)

    print("\n" + "-" * 20)
    print(f"  Volumetric Attr ({args.method})")
    if "roi_ratio" in res.columns:
        print(res.groupby(["label", "correct"]).agg(
            n=("roi_ratio", "size"), roi_ratio=("roi_ratio", "mean"),
            sd=("roi_ratio", "std")).round(3).to_string())
        y, o = res[res.age < 70], res[res.age >= 80]
        if len(y) >= 10 and len(o) >= 10:
            print(f"\nunder 70 (n={len(y)}) {y.roi_ratio.mean():.3f}   "
                  f"over 80 (n={len(o)}) {o.roi_ratio.mean():.3f}   "
                  f"difference {o.roi_ratio.mean() - y.roi_ratio.mean():+.3f}")
    if sanity:
        print(f"\nsanity check correlation: {sanity[0]:.3f}")
    print(f"\nWrote {out} and {len(list(figs.glob('*.png')))} figures in {figs}")


if __name__ == "__main__":
    main()
