import os
import argparse
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _normpath(p: str) -> str:
    return os.path.normpath(str(p))


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=os.path.join(DATA_DIR, "adni_metadata.csv"))
    ap.add_argument("--out_root", default=os.path.join(DATA_DIR, "splits"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    return ap


def main():
    args = build_argparser().parse_args()

    if not os.path.exists(args.metadata):
        raise RuntimeError(f"Missing metadata: {args.metadata}")

    df = pd.read_csv(args.metadata)

    required = ["subject_id", "scan_path"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"metadata missing column: {c}")

    # TEMPORARY LABEL FIXXXXXXXXXXX
    # Single dummy class so pipeline runs
    df["label"] = 0

    df["scan_path"] = df["scan_path"].astype(str).map(_normpath)
    df = df.sort_values("subject_id").groupby("subject_id", as_index=False).first()

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_test = int(len(df) * args.test_frac)
    n_val = int(len(df) * args.val_frac)

    test = df.iloc[idx[:n_test]]
    val = df.iloc[idx[n_test:n_test + n_val]]
    train = df.iloc[idx[n_test + n_val:]]

    out_dir = os.path.join(args.out_root, "adni_dummy")
    os.makedirs(out_dir, exist_ok=True)
    vdir = os.path.join(out_dir, "validated")
    os.makedirs(vdir, exist_ok=True)

    train.to_csv(os.path.join(vdir, "train.csv"), index=False)
    val.to_csv(os.path.join(vdir, "val.csv"), index=False)
    test.to_csv(os.path.join(vdir, "test.csv"), index=False)

    print("Dummy splits written:")
    print("train:", len(train))
    print("val:", len(val))
    print("test:", len(test))


if __name__ == "__main__":
    main()
