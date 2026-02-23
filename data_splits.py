import os
import re
import argparse
from typing import Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except Exception as e:
    StratifiedShuffleSplit = None

# Changeable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _normpath(p: str) -> str:
    return os.path.normpath(str(p))


def _exists_nonempty(p: str) -> bool:
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except OSError:
        return False


def _pick_one_scan_per_subject(df: pd.DataFrame, prefer_contains: Tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    df["scan_path"] = df["scan_path"].astype(str).map(_normpath)
    df["scan_filename"] = df.get("scan_filename", df["scan_path"].map(lambda x: os.path.basename(x))).astype(str)

    # scoring for "best" scan per subject
    def score_row(r) -> int:
        s = 0
        fn = str(r["scan_filename"]).lower()
        sp = str(r["scan_path"]).lower()
        for tok in prefer_contains:
            if tok and tok.lower() in fn:
                s += 3
            if tok and tok.lower() in sp:
                s += 1
        if "masked" in fn or "masked" in sp:
            s += 1
        return s

    df["_score"] = df.apply(score_row, axis=1)
    # stable order: higher score first, then path
    df = df.sort_values(["subject_id", "_score", "scan_path"], ascending=[True, False, True])
    df = df.groupby("subject_id", as_index=False).first()
    df = df.drop(columns=["_score"])
    return df


def _stratified_split_subjects(subject_df: pd.DataFrame, seed: int, val_frac: float, test_frac: float):
    if StratifiedShuffleSplit is None:
        raise RuntimeError("scikit-learn missing: install scikit-learn")

    y = subject_df["label"].astype(int).to_numpy()
    idx = np.arange(len(subject_df))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx, y))

    trainval = subject_df.iloc[trainval_idx].reset_index(drop=True)
    test = subject_df.iloc[test_idx].reset_index(drop=True)

    # val fraction relative to train-val
    val_rel = val_frac / max(1e-8, (1.0 - test_frac))
    y_tv = trainval["label"].astype(int).to_numpy()
    idx_tv = np.arange(len(trainval))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_idx, val_idx = next(sss2.split(idx_tv, y_tv))

    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def _validate_and_write(df: pd.DataFrame, out_csv: str) -> Tuple[int, int, int]:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = df.copy()
    df["scan_path"] = df["scan_path"].astype(str).map(_normpath)

    ok = df["scan_path"].map(_exists_nonempty)
    good = df[ok].copy()
    bad = df[~ok].copy()

    good.to_csv(out_csv, index=False)

    return int(len(df)), int(len(good)), int(len(bad))


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(DATA_DIR, "metadata.csv"))
    ap.add_argument("--out_root", default=os.path.join(DATA_DIR, "splits"))
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--only_mr1", action="store_true", help="Keep only session_id ending with _MR1")
    ap.add_argument("--binary", action="store_true", help="Map label: 0 stays 0, everything else -> 1")
    ap.add_argument("--prefer", default="mpr-1,mpr_1", help="Comma tokens to prefer when picking 1 scan per subject")
    return ap


def main():
    args = build_argparser().parse_args()

    if not os.path.exists(args.metadata):
        raise RuntimeError(f"Missing metadata.csv: {args.metadata}")

    df = pd.read_csv(args.metadata)
    for col in ["subject_id", "session_id", "resolution", "scan_path", "label"]:
        if col not in df.columns:
            raise RuntimeError(f"metadata.csv missing column: {col}")

    df["scan_path"] = df["scan_path"].astype(str).map(_normpath)
    df["resolution"] = df["resolution"].astype(int)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)

    df = df[df["resolution"] == int(args.resolution)].copy()

    if args.only_mr1:
        df = df[df["session_id"].astype(str).str.contains(r"_MR1$", flags=re.IGNORECASE, regex=True)].copy()

    if args.binary:
        df["label"] = (df["label"] != 0).astype(int)

    # scan per subject to avoid leakage and inflated counts
    prefer = tuple([t.strip() for t in str(args.prefer).split(",") if t.strip()])
    df_subj = _pick_one_scan_per_subject(df, prefer_contains=prefer)

    # create output folder name consistent with runs
    tag = f"res_{int(args.resolution)}"
    if args.binary:
        tag += "_binary"
    if args.only_mr1:
        tag += "_mr1"
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    train, val, test = _stratified_split_subjects(df_subj, seed=int(args.seed), val_frac=float(args.val_frac), test_frac=float(args.test_frac))

    # split label
    train = train.copy(); train["split"] = "train"
    val = val.copy(); val["split"] = "val"
    test = test.copy(); test["split"] = "test"

    # write raw splits
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    # validated splits
    vdir = os.path.join(out_dir, "validated")
    os.makedirs(vdir, exist_ok=True)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        n_in, n_good, n_bad = _validate_and_write(part, os.path.join(vdir, f"{name}.csv"))
        print(f"{name}: in={n_in} good={n_good} bad={n_bad}")

    print(f"Wrote: {out_dir}")
    print(f"Wrote validated splits to: {vdir}")

    # print counts
    def subj_counts(d: pd.DataFrame):
        return int(d["subject_id"].nunique())

    print("Subjects:", {"train": subj_counts(train), "val": subj_counts(val), "test": subj_counts(test)})
    print("Train labels:\n", train["label"].value_counts().sort_index().to_string())
    print("Val labels:\n", val["label"].value_counts().sort_index().to_string())
    print("Test labels:\n", test["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
