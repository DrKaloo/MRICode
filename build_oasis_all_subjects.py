"""
Build OASIS all-subjects test CSV from the canonical 128^3 3-class splits.

Only picks files matching the dissertation's main pipeline:
    - data/splits/res_128/5fold/fold_{1..5}/validated/test.csv  (5-fold CV)
    - data/splits/res_128/validated/test.csv                    (global holdout)

Ignores: binary-classification variants, lower-resolution variants, and
unvalidated (non-/validated/) test CSVs.

Run from your Msc-AD/ directory:
    python build_oasis_all_subjects.py
"""

import sys
from pathlib import Path
import pandas as pd

splits_root = Path("data/splits/res_128")
if not splits_root.is_dir():
    sys.exit(f"splits root not found: {splits_root.resolve()}")

test_files = []

# 5-fold validated tests (1-indexed, fold_1 to fold_5)
for fold in range(1, 6):
    p = splits_root / "5fold" / f"fold_{fold}" / "validated" / "test.csv"
    if p.is_file():
        test_files.append(p)
    else:
        print(f"[Warn] Missing fold {fold}: {p}")

# Global holdout - try common locations
holdout_candidates = [
    splits_root / "validated" / "test.csv",
    splits_root / "holdout" / "validated" / "test.csv",
]
holdout_found = None
for c in holdout_candidates:
    if c.is_file():
        holdout_found = c
        break

if holdout_found:
    test_files.append(holdout_found)
    print(f"[Holdout] {holdout_found}")
else:
    print("[Warn] no holdout CSV found; using only 5-fold tests")

if not test_files:
    sys.exit("no canonical test CSVs found under data/splits/res_128/")

print(f"[Found] {len(test_files)} test CSVs:")
for f in test_files:
    print(f"   - {f}")

dfs = [pd.read_csv(f) for f in test_files]
combined = pd.concat(dfs, ignore_index=True)

n_pre = len(combined)
combined = combined.drop_duplicates(subset=["subject_id"], keep="first")
n_post = len(combined)
if n_pre == n_post:
    print(f"[OK]    {n_post} unique subjects (no duplicates)")
else:
    print(f"[Dedup] {n_pre} rows -> {n_post} unique subjects")

print(f"[Class balance] {combined['label'].value_counts().sort_index().to_dict()}")

out = splits_root / "all" / "validated" / "test.csv"
out.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out, index=False)
print(f"\n[Done] wrote {out}")