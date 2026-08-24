import os
import re
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

except Exception as e:
    StratifiedShuffleSplit = None
    StratifiedKFold = None

# Changeable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


#     Path helpers

def _normpath(p: str) -> str:
    return os.path.normpath(str(p))


def _exists_nonempty(p: str) -> bool:
    try:
        return os.path.exists(p) and os.path.getsize(p) > 0
    except OSError:
        return False


#  Scan selection

def _pick_one_scan_per_subject(df: pd.DataFrame, prefer_contains: Tuple[str, ...]) -> pd.DataFrame:
    df = df.copy()
    df["scan_path"] = df["scan_path"].astype(str).map(_normpath)
    df["scan_filename"] = df.get("scan_filename", df["scan_path"].map(lambda x: os.path.basename(x))).astype(str)

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
    df = df.sort_values(["subject_id", "_score", "scan_path"], ascending=[True, False, True])
    df = df.groupby("subject_id", as_index=False).first()
    df = df.drop(columns=["_score"])
    return df


# Age-bin assignment

def _assign_age_bins(df: pd.DataFrame, bin_edges: List[float]) -> pd.DataFrame:
    """
    Adds 'age_bin' column based on bin edges.
    Subjects with missing age are assigned age_bin = -1.

    Example: bin_edges=[0, 70, 80, 200] creates bins [0,70), [70,80), [80,200]
             labelled as 0, 1, 2 respectively.
    """
    df = df.copy()
    ages = pd.to_numeric(df["age"], errors="coerce")

    labels_for_bins = list(range(len(bin_edges) - 1))
    df["age_bin"] = pd.cut(ages, bins=bin_edges, right=False, labels=labels_for_bins)
    df["age_bin"] = df["age_bin"].cat.add_categories([-1]).fillna(-1).astype(int)

    return df


# Age-bin-wise class balancing

def _balance_within_age_bins(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Within each age bin, undersample all classes to match the smallest class
    in that bin. This ensures equal class representation per age group.

    Only applied to training data. Val/test are never touched.

    Returns the balanced dataframe plus prints a report of what was removed.
    """
    rng = np.random.RandomState(seed)
    balanced_parts = []

    for ab, bin_df in df.groupby("age_bin"):
        class_counts = bin_df["label"].value_counts()
        target_n = int(class_counts.min())  # match the smallest class in this bin

        if target_n == 0:
            print(f"[WARNING] age_bin={ab} has a class with 0 subjects, skipping entire bin")
            continue

        for lab, class_df in bin_df.groupby("label"):
            if len(class_df) <= target_n:
                balanced_parts.append(class_df)
            else:
                balanced_parts.append(class_df.sample(n=target_n, random_state=rng))

    if not balanced_parts:
        raise RuntimeError("Balancing produced an empty training set. Check age/label distributions.")

    result = pd.concat(balanced_parts, ignore_index=True)

    # Report
    n_before = len(df)
    n_after = len(result)
    print(f"  Age-bin balancing: {n_before} -> {n_after} subjects ({n_before - n_after} removed)")
    for ab in sorted(result["age_bin"].unique()):
        sub = result[result["age_bin"] == ab]
        counts_str = sub["label"].value_counts().sort_index().to_dict()
        print(f"  bin={ab}: {counts_str} (n={len(sub)})")

    return result


# Stratified split

def _stratified_split_subjects(subject_df: pd.DataFrame, seed: int, val_frac: float, test_frac: float):
    if StratifiedShuffleSplit is None:
        raise RuntimeError("scikit-learn missing: install scikit-learn")

    y = subject_df["label"].astype(int).to_numpy()
    idx = np.arange(len(subject_df))

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx, y))

    trainval = subject_df.iloc[trainval_idx].reset_index(drop=True)
    test = subject_df.iloc[test_idx].reset_index(drop=True)

    val_rel = val_frac / max(1e-8, (1.0 - test_frac))
    y_tv = trainval["label"].astype(int).to_numpy()
    idx_tv = np.arange(len(trainval))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_idx, val_idx = next(sss2.split(idx_tv, y_tv))

    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)
    return train, val, test


# Validation set from a train+val pool

def _carve_val_from_trainval(trainval_df: pd.DataFrame, val_frac: float, seed: int, strat_col: str = "label"):
    """
    Given a combined train+val dataframe, split off a validation subset.
    Stratified by strat_col (defaults to label, can be composite key).
    """
    if StratifiedShuffleSplit is None:
        raise RuntimeError("scikit-learn missing")

    y = trainval_df[strat_col].to_numpy()
    idx = np.arange(len(trainval_df))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(sss.split(idx, y))

    train = trainval_df.iloc[train_idx].reset_index(drop=True)
    val = trainval_df.iloc[val_idx].reset_index(drop=True)
    return train, val


# K-fold splittingggggggggggggg

def _kfold_split_subjects(subject_df: pd.DataFrame, n_folds: int, seed: int, val_frac: float):
    """
    Outer K-fold: split all subjects into K folds (each fold's test set is one partition).
    Inner: within each fold's train+val pool, carve out a val set.

    Stratification uses a composite key of (age_bin, label) so that both
    age and diagnosis distributions are preserved across folds.
    """
    if StratifiedKFold is None:
        raise RuntimeError("scikit-learn missing: install scikit-learn")

    df = subject_df.copy()

    # Composite stratification key: ensures each fold has similar
    # age_bin x label proportions (not just label alone)
    df["_strat_key"] = df["age_bin"].astype(str) + "_" + df["label"].astype(str)

    # If any group has fewer members than n_folds, fall back to label-only stratification
    min_group = df["_strat_key"].value_counts().min()
    if min_group < n_folds:
        print(f"  [INFO] Smallest (age_bin x label) group has {min_group} subjects < {n_folds} folds.")
        print(f"         Falling back to label-only stratification for fold assignment.")
        df["_strat_key"] = df["label"].astype(str)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_strat = df["_strat_key"].to_numpy()

    folds = []
    for fold_i, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), y_strat)):
        trainval = df.iloc[trainval_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        # val from trainval, stratified by label
        train, val = _carve_val_from_trainval(trainval, val_frac=val_frac, seed=seed + fold_i + 1)

        # Clean up helper column
        for part in [train, val, test]:
            if "_strat_key" in part.columns:
                part.drop(columns=["_strat_key"], inplace=True)

        folds.append((train, val, test))

    return folds


# Validation and writing

def _validate_and_write(df: pd.DataFrame, out_csv: str) -> Tuple[int, int, int]:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = df.copy()
    df["scan_path"] = df["scan_path"].astype(str).map(_normpath)

    ok = df["scan_path"].map(_exists_nonempty)
    good = df[ok].copy()
    bad = df[~ok].copy()

    good.to_csv(out_csv, index=False)

    return int(len(df)), int(len(good)), int(len(bad))


# Args

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

    # K-fold
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds (1 = legacy single split)")

    # Age-bin balancing
    ap.add_argument("--age_bins", default="0,70,80,200",
                     help="Comma-separated bin edges for age groups (default: 0,70,80,200 -> <70, 70-80, 80+)")
    ap.add_argument("--balance_age_bins", action="store_true",
                     help="Undersample training set so classes are balanced within each age bin")

    return ap


# Main

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

    # One scan per subject to avoid leakage ******************************************************************
    prefer = tuple([t.strip() for t in str(args.prefer).split(",") if t.strip()])
    df_subj = _pick_one_scan_per_subject(df, prefer_contains=prefer)

    # Assign age bins to all subjects before splitting *********************************
    bin_edges = [float(x.strip()) for x in args.age_bins.split(",")]
    df_subj = _assign_age_bins(df_subj, bin_edges=bin_edges)

    bin_labels = [f"[{bin_edges[i]},{bin_edges[i+1]})" for i in range(len(bin_edges) - 1)]
    print(f"Age bins: {bin_labels}")
    print("Age-bin distribution before splitting:")
    for ab in sorted(df_subj["age_bin"].unique()):
        sub = df_subj[df_subj["age_bin"] == ab]
        tag_name = f"  bin {ab}" if ab >= 0 else "  missing age"
        print(f"{tag_name}: n={len(sub)}  labels={sub['label'].value_counts().sort_index().to_dict()}")

    # Output folder tag
    tag = f"res_{int(args.resolution)}"
    if args.binary:
        tag += "_binary"
    if args.only_mr1:
        tag += "_mr1"

    n_folds = int(args.n_folds)

    # K-fold vs single split (starter version)

    if n_folds > 1:
        # K-FOLD PATH
        print(f"\n=== {n_folds}-Fold Cross-Validation ===\n")

        folds = _kfold_split_subjects(df_subj, n_folds=n_folds, seed=int(args.seed), val_frac=float(args.val_frac))

        for fold_i, (train, val, test) in enumerate(folds):
            fold_tag = f"fold_{fold_i + 1}"
            fold_dir = os.path.join(args.out_root, tag, f"{n_folds}fold", fold_tag)
            os.makedirs(fold_dir, exist_ok=True)

            print(f"\n--- Fold {fold_i + 1}/{n_folds} ---")
            print(f"  Before balancing: train={len(train)} val={len(val)} test={len(test)}")

            # Age-bin balancing on training set only
            if args.balance_age_bins:
                train = _balance_within_age_bins(train, seed=int(args.seed) + fold_i)
                print(f"  After balancing:  train={len(train)}")

            train = train.copy(); train["split"] = "train"
            val = val.copy(); val["split"] = "val"
            test = test.copy(); test["split"] = "test"

            # Write raw splits
            train.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
            val.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
            test.to_csv(os.path.join(fold_dir, "test.csv"), index=False)

            # Validated splits
            vdir = os.path.join(fold_dir, "validated")
            os.makedirs(vdir, exist_ok=True)
            for name, part in [("train", train), ("val", val), ("test", test)]:
                n_in, n_good, n_bad = _validate_and_write(part, os.path.join(vdir, f"{name}.csv"))
                print(f"  {name}: in={n_in} good={n_good} bad={n_bad}")

            # Per-fold summary
            print(f"  Train labels: {train['label'].value_counts().sort_index().to_dict()}")
            print(f"  Val labels:   {val['label'].value_counts().sort_index().to_dict()}")
            print(f"  Test labels:  {test['label'].value_counts().sort_index().to_dict()}")

        print(f"\nWrote {n_folds} folds to: {os.path.join(args.out_root, tag, f'{n_folds}fold')}")

    else:
        # Original "SINGLE-SPLIT PATH"
        print("\n=== Single Split (legacy mode, --n_folds=1) ===\n")

        out_dir = os.path.join(args.out_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        train, val, test = _stratified_split_subjects(
            df_subj, seed=int(args.seed),
            val_frac=float(args.val_frac), test_frac=float(args.test_frac),
        )

        print(f"Before balancing: train={len(train)} val={len(val)} test={len(test)}")

        # Age-bin balancing on training set only
        if args.balance_age_bins:
            train = _balance_within_age_bins(train, seed=int(args.seed))
            print(f"After balancing:  train={len(train)}")

        train = train.copy(); train["split"] = "train"
        val = val.copy(); val["split"] = "val"
        test = test.copy(); test["split"] = "test"

        train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val.to_csv(os.path.join(out_dir, "val.csv"), index=False)
        test.to_csv(os.path.join(out_dir, "test.csv"), index=False)

        vdir = os.path.join(out_dir, "validated")
        os.makedirs(vdir, exist_ok=True)
        for name, part in [("train", train), ("val", val), ("test", test)]:
            n_in, n_good, n_bad = _validate_and_write(part, os.path.join(vdir, f"{name}.csv"))
            print(f"{name}: in={n_in} good={n_good} bad={n_bad}")

        print(f"Wrote: {out_dir}")
        print(f"Wrote validated splits to: {vdir}")

        def subj_counts(d: pd.DataFrame):
            return int(d["subject_id"].nunique())

        print("Subjects:",
              {"train": subj_counts(train), "val": subj_counts(val), "test": subj_counts(test)})
        print("Train labels:"
              "\n", train["label"].value_counts().sort_index().to_string())
        print("Val labels:"
              "\n", val["label"].value_counts().sort_index().to_string())
        print("Test labels:"
              "\n", test["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()














