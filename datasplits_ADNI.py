"""
ADNI dataset splitting, mirrors data_splits.py for OASIS.

Structure produced (matches OASIS so train.py / evaluate.py work unchanged):

    {out_root}/
        all/validated/test.csv             (optional: every eligible subject,
                                            for OASIS->ADNI cross-cohort eval)
        holdout/validated/test.csv         (20% global held-out test, for
                                            unbiased final reporting)
        fold_0/validated/
            train.csv                      (~72% of the 80% pool, age-bin balanced)
            val.csv                        (~8% of the 80% pool, early stopping)
            test.csv                       (the CV fold's held-out 1/5 of pool)
        fold_1/validated/...
        ...
        fold_4/validated/...

Stratification key: composite age_bin x label, falls back to label only when
any composite group has fewer subjects than n_folds. Within each fold the
training portion is age-bin balanced (each class undersampled to the smallest
class within the bin); val/test left at natural distribution for honest eval.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def assign_age_bin(age: float) -> int:
    """Bin 0: <70, Bin 1: 70-80, Bin 2: 80+."""
    if pd.isna(age):
        return -1
    if age < 70:
        return 0
    if age < 80:
        return 1
    return 2


def make_strat_key(df: pd.DataFrame, n_min: int) -> pd.Series:
    """Composite age_bin x label; falls back to label-only if any group < n_min."""
    composite = df["age_bin"].astype(str) + "_" + df["label"].astype(str)
    if composite.value_counts().min() < n_min:
        return df["label"].astype(str)
    return composite


def balance_train_by_age_bin(
    train_df: pd.DataFrame, rng: np.random.RandomState
) -> pd.DataFrame:
    """Within each age bin, undersample classes to the smallest class size."""
    chunks = []
    for _, bin_df in train_df.groupby("age_bin"):
        if len(bin_df) == 0:
            continue
        cls_counts = bin_df["label"].value_counts()
        min_count = int(cls_counts.min())
        for _, cls_df in bin_df.groupby("label"):
            if len(cls_df) > min_count:
                cls_df = cls_df.sample(
                    n=min_count, random_state=rng.randint(0, 2**31 - 1)
                )
            chunks.append(cls_df)
    if not chunks:
        return train_df.iloc[0:0].copy()
    out = pd.concat(chunks, axis=0).sample(
        frac=1, random_state=rng.randint(0, 2**31 - 1)
    )
    return out.reset_index(drop=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        default=str(DATA_DIR / "adni_metadata.csv"),
        help="Merged metadata CSV with subject_id, scan_path, label, age, sex.",
    )
    ap.add_argument(
        "--out_root",
        default=str(DATA_DIR / "splits_adni"),
        help="Where to write fold directories.",
    )
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac_in_train", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--no_balance",
        action="store_true",
        help="Skip age-bin training balance (useful for ablation comparison).",
    )
    ap.add_argument(
        "--write_all_subjects",
        action="store_true",
        help="Also write splits_adni/all/validated/test.csv with every subject "
             "(for OASIS->ADNI cross-cohort eval where ADNI is one big test set).",
    )
    return ap


def main():
    args = build_argparser().parse_args()

    if not os.path.exists(args.metadata):
        raise SystemExit(f"missing metadata: {args.metadata}")

    df = pd.read_csv(args.metadata)

    required = ["subject_id", "scan_path", "label", "age"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"metadata missing required columns: {missing}")

    n_pre = len(df)
    df = df.dropna(subset=["label", "age"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["age"] = df["age"].astype(float)
    df["age_bin"] = df["age"].apply(assign_age_bin)
    df = df[df["age_bin"] >= 0].reset_index(drop=True)
    print(f"[Filter] {n_pre} -> {len(df)} subjects (dropped missing label/age)")
    print(f"[Class balance]    {df['label'].value_counts().sort_index().to_dict()}")
    print(f"[Age bin balance]  {df['age_bin'].value_counts().sort_index().to_dict()}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    # "All subjects" CSV for cross-cohort eval (no train/val/test split)
    if args.write_all_subjects:
        all_dir = out_root / "all" / "validated"
        all_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(all_dir / "test.csv", index=False)
        print(f"[All] wrote {len(df)} subjects to {all_dir / 'test.csv'}")

    # 20% global held-out test (final unbiased eval)
    strat_global = make_strat_key(df, n_min=2)
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_frac, random_state=args.seed
    )
    pool_idx, holdout_idx = next(sss.split(df, strat_global))
    pool_df = df.iloc[pool_idx].reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)

    holdout_dir = out_root / "holdout" / "validated"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(holdout_dir / "test.csv", index=False)
    print(f"[Holdout] pool={len(pool_df)}  holdout_test={len(holdout_df)}")

    # 5-fold stratified CV on pool
    pool_strat = make_strat_key(pool_df, n_min=args.n_folds)
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.seed
    )

    for fold_idx, (train_pool_idx, fold_test_idx) in enumerate(
        skf.split(pool_df, pool_strat)
    ):
        fold_dir = out_root / f"fold_{fold_idx}" / "validated"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train_pool = pool_df.iloc[train_pool_idx].reset_index(drop=True)
        fold_test = pool_df.iloc[fold_test_idx].reset_index(drop=True)

        # 90/10 inner split for early stopping; fold-specific seed prevents
        # correlated val sets across folds (matches OASIS handoff note)
        inner_strat = make_strat_key(fold_train_pool, n_min=2)
        sss_inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.val_frac_in_train,
            random_state=args.seed + fold_idx,
        )
        train_idx, dev_idx = next(sss_inner.split(fold_train_pool, inner_strat))
        fold_train = fold_train_pool.iloc[train_idx].reset_index(drop=True)
        fold_dev = fold_train_pool.iloc[dev_idx].reset_index(drop=True)

        if not args.no_balance:
            fold_train = balance_train_by_age_bin(fold_train, rng)

        fold_train.to_csv(fold_dir / "train.csv", index=False)
        fold_dev.to_csv(fold_dir / "val.csv", index=False)
        fold_test.to_csv(fold_dir / "test.csv", index=False)

        print(
            f"[Fold {fold_idx}] train={len(fold_train)}  "
            f"val={len(fold_dev)}  test(CV)={len(fold_test)}"
        )

    print(f"[Done] splits written under {out_root}")


if __name__ == "__main__":
    main()
