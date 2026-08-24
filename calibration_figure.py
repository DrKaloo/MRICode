"""
    python calibration_figure.py --index analysis_out\\run_index.csv
    python calibration_figure.py --index analysis_out\\run_index.csv ^
        --figure_cell 2class r3d_18 ADNI ADNI
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ece_binned(conf, correct, n_bins=10):
    """Size-weighted mean of | accuracy - confidence | within confidence bins.
    Returns (ece, per-bin table) with empty bins dropped."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if conf.size == 0:
        return np.nan, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        total += (m.sum() / conf.size) * abs(acc - cf)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                     "confidence": cf, "accuracy": acc})
    return float(total), rows


def load_fold(path, k):
    df = pd.read_csv(path)
    cols = [f"prob_{i}" for i in range(k)]
    P = df[cols].to_numpy()
    y = df["label"].to_numpy().astype(int)
    pred = df["pred"].to_numpy().astype(int) if "pred" in df.columns else P.argmax(1)
    if "age_bin" not in df.columns:
        df["age_bin"] = pd.cut(df["age"], [0, 70, 80, 200], right=False,
                               labels=[0, 1, 2]).astype(float)
    return pd.DataFrame({
        "subject_id": df.get("subject_id", pd.Series(range(len(df)))).astype(str),
        "age_bin": df["age_bin"],
        "confidence": P.max(axis=1),
        "correct": (pred == y).astype(int),
        "label": y, "pred": pred,
    })


BIN_NAMES = {0: "under 70", 1: "70 to 80", 2: "over 80"}


def make_figure(frames, direction, title, out_png, n_bins=10):
    pooled = (pd.concat(frames, ignore_index=True).drop_duplicates("subject_id")
              if direction == "within" else pd.concat(frames, ignore_index=True))

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    ax = axes[0, 0]
    e, rows = ece_binned(pooled["confidence"], pooled["correct"], n_bins)
    if rows:
        t = pd.DataFrame(rows)
        ax.plot(t["confidence"], t["accuracy"], "s-", lw=2, ms=8, label="model")
        for _, r in t.iterrows():
            ax.annotate(f"n={r['n']:.0f}", (r["confidence"], r["accuracy"]),
                        fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"Overall reliability\nECE = {e:.3f}", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[0, 1]
    per_bin = {}
    for b, colr in zip([0, 1, 2], ["tab:green", "tab:orange", "tab:red"]):
        sub = pooled[pooled["age_bin"] == b]
        if len(sub) < 10:
            continue
        eb, rb = ece_binned(sub["confidence"], sub["correct"], max(4, n_bins // 2))
        per_bin[b] = (eb, len(sub), sub["confidence"].mean(), sub["correct"].mean())
        if rb:
            t = pd.DataFrame(rb)
            ax.plot(t["confidence"], t["accuracy"], "o-", lw=2, ms=7, color=colr,
                    label=f"{BIN_NAMES[b]} (n={len(sub)}, ECE {eb:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability by age bin", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax = axes[1, 0]
    for b, colr in zip([0, 1, 2], ["tab:green", "tab:orange", "tab:red"]):
        sub = pooled[pooled["age_bin"] == b]
        if len(sub):
            ax.hist(sub["confidence"], bins=12, alpha=0.5, color=colr,
                    edgecolor="black", label=f"{BIN_NAMES[b]} (n={len(sub)})")
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("count")
    ax.set_title("Confidence distribution by age bin", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    ax.axis("off")
    lines = [f"Overall ECE: {e:.3f}   (n = {len(pooled)})", ""]
    for b in sorted(per_bin):
        eb, n, mc, ma = per_bin[b]
        lines += [f"{BIN_NAMES[b]}  (n = {n})",
                  f"   ECE                {eb:.3f}",
                  f"   mean confidence    {mc:.3f}",
                  f"   observed accuracy  {ma:.3f}",
                  f"   {'over' if mc > ma else 'under'}confident by {abs(mc - ma):.3f}",
                  ""]
    if len(per_bin) >= 2:
        vals = {b: v[0] for b, v in per_bin.items()}
        w, bst = max(vals, key=vals.get), min(vals, key=vals.get)
        lines += [f"Worst-minus-best ECE gap: {vals[w] - vals[bst]:.3f}",
                  f"   worst: {BIN_NAMES[w]}   best: {BIN_NAMES[bst]}"]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            va="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return e, per_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="analysis_out/run_index.csv")
    ap.add_argument("--outdir", default="analysis_out/calibration")
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument("--figure_cell", nargs=4, default=None,
                    metavar=("ARM", "ARCH", "TRAIN", "EVAL"),
                    help="Only make a figure for this cell. Default: every cell.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(args.index)
    idx = idx[idx["arm"].isin(["2class", "3class"])]
    idx = idx[idx["train_cohort"].isin(["OASIS", "ADNI"])
              & idx["eval_cohort"].isin(["OASIS", "ADNI"])]

    rows = []
    for (arm, arch, tr, ev), g in idx.groupby(
            ["arm", "arch", "train_cohort", "eval_cohort"]):
        k = 2 if arm == "2class" else 3
        direction = "within" if tr == ev else "cross"
        frames = [load_fold(p, k) for p in g.sort_values("fold")["pred_file"]]

        # per-fold ECE, then averaged. Each fold has its own probability scale,
        # so this is the honest aggregate; the figure pools for readability.
        fold_ece = [ece_binned(f["confidence"], f["correct"], args.n_bins)[0]
                    for f in frames]

        want = (args.figure_cell is None or
                list(args.figure_cell) == [arm, arch, tr, ev])
        e_pool, per_bin = (np.nan, {})
        if want:
            name = f"{arm}_{arch}_{tr}_to_{ev}".replace("/", "_")
            title = (f"Calibration: {arch}, {arm}, trained on {tr}, "
                     f"evaluated on {ev} ({direction})")
            e_pool, per_bin = make_figure(frames, direction, title,
                                          outdir / f"{name}.png", args.n_bins)
            print(f"  wrote {outdir / (name + '.png')}")

        row = {"arm": arm, "arch": arch, "train": tr, "eval": ev,
               "direction": direction, "n_folds": len(frames),
               "ece_mean_of_folds": round(float(np.nanmean(fold_ece)), 4),
               "ece_sd_of_folds": round(float(np.nanstd(fold_ece, ddof=1)), 4)
               if len(fold_ece) > 1 else None,
               "ece_pooled": round(e_pool, 4) if e_pool == e_pool else None}
        for b in [0, 1, 2]:
            row[f"ece_bin{b}"] = round(per_bin[b][0], 4) if b in per_bin else None
        if len(per_bin) >= 2:
            v = [per_bin[b][0] for b in per_bin]
            row["ece_age_gap"] = round(max(v) - min(v), 4)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["arm", "direction", "train", "arch"])
    out.to_csv(outdir / "calibration_summary.csv", index=False)

    print("\n" + "-" * 20)
    print("4.8  Calibration, BINNED Expected Calibration Error")


    print(out.to_string(index=False))
    print("\nECE 0 is perfect. Higher is worse. Because these models produce")
    print("probabilities squashed into a narrow band, expect confidence to sit close")
    print("to the base rate and ECE to be driven by that rather than by sharpness.")
    print("Compare ece_age_gap against the per-bin temperature-scaling result already")
    print("reported: MobileNet 0.130 to 0.041, MedicalNet 0.120 to 0.052, R3D-18 0.090")
    print("to 0.153. These are now computed the same way and are comparable.")
    print(f"\nWrote {outdir / 'calibration_summary.csv'}")


if __name__ == "__main__":
    main()
