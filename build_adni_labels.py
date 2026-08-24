import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


ADNIMERGE2_DIR = Path(r"C:\Users\todor\Desktop\ADNIMERGE2\data")
DXSUM_DEFAULT = ADNIMERGE2_DIR / "DXSUM.rda"
PTDEMOG_DEFAULT = ADNIMERGE2_DIR / "PTDEMOG.rda"

# Map DXSUM.DIAGNOSIS values to integer (Category integer) labels.
# Modern ADNI uses string values; ADNI1 historically used numeric codes (1/2/3).
DIAGNOSIS_STR_TO_LABEL = {
    "CN":       0,
    "NORMAL":   0,
    "NL":       0,
    "MCI":      1,
    "EMCI":     1,
    "LMCI":     1,
    "DEMENTIA": 2,
    "AD":       2,
    "DEM":      2,
}

# Numeric codes used in ADNI1 DXCURREN; sometimes leak into DIAGNOSIS as ints.
DIAGNOSIS_NUM_TO_LABEL = {1: 0, 2: 1, 3: 2}


def read_table(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() in (".rda", ".rdata"):
        import pyreadr  # only needed for the R-format branch

        result = pyreadr.read_r(path)
        if not result:
            raise SystemExit(f"No table found inside: {path}")
        return next(iter(result.values()))
    return pd.read_csv(path, low_memory=False)


def map_diagnosis(v: object) -> Optional[int]:
    """Map a DXSUM DIAGNOSIS cell to {0,1,2} or None."""
    if pd.isna(v):
        return None

    s = str(v).strip().upper()
    if s in DIAGNOSIS_STR_TO_LABEL:
        return DIAGNOSIS_STR_TO_LABEL[s]

    try:
        n = int(float(s))
        if n in DIAGNOSIS_NUM_TO_LABEL:
            return DIAGNOSIS_NUM_TO_LABEL[n]
    except (ValueError, TypeError):
        pass
    return None


# PTDOB format: "MM/YYYY" (month and year only, day suppressed for privacy).
PTDOB_RE = re.compile(r"^(\d{1,2})/(\d{4})$")


def parse_ptdob(ptdob_str: object) -> Optional[pd.Timestamp]:
    """Parse PTDOB ('MM/YYYY') to Timestamp at the 15th of given month."""
    if not isinstance(ptdob_str, str):
        return None
    m = PTDOB_RE.match(ptdob_str.strip())
    if not m:
        return None
    month, year = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12) or not (1900 <= year <= 2025):
        return None
    return pd.Timestamp(year=year, month=month, day=15)


def compute_age(ptdob: Optional[pd.Timestamp], examdate: object) -> Optional[float]:
    """Age in years from birth to a reference date, rounded to 1 decimal."""
    if ptdob is None or pd.isna(examdate):
        return None
    try:
        ref = pd.Timestamp(examdate)
    except (ValueError, TypeError):
        return None
    age_days = (ref - ptdob).days
    if age_days < 0 or age_days > 150 * 365:
        return None
    return round(age_days / 365.25, 1)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_metadata",
        default=str(DATA_DIR / "adni_raw_metadata.csv"),
        help="Output of preprocess_ADNI.py",
    )
    ap.add_argument(
        "--dxsum",
        default=str(DXSUM_DEFAULT),
        help="Path to DXSUM (.rda or .csv)",
    )
    ap.add_argument(
        "--ptdemog",
        default=str(PTDEMOG_DEFAULT),
        help="Path to PTDEMOG (.rda or .csv)",
    )
    ap.add_argument(
        "--out",
        default=str(DATA_DIR / "adni_metadata.csv"),
    )
    ap.add_argument(
        "--allow_missing_labels",
        action="store_true",
    )
    return ap


def main():
    args = build_argparser().parse_args()

    for label, p in [("raw_metadata", args.raw_metadata),
                     ("dxsum", args.dxsum),
                     ("ptdemog", args.ptdemog)]:
        if not Path(p).is_file():
            raise SystemExit(f"{label} not found: {p}")

    raw = read_table(args.raw_metadata)
    dxsum = read_table(args.dxsum)
    ptdemog = read_table(args.ptdemog)

    print(f"[Load] raw scans: {len(raw)}")
    print(f"[Load] DXSUM rows: {len(dxsum)}")
    print(f"[Load] PTDEMOG rows: {len(ptdemog)}")

    # ----------------- Baseline diagnosis from DXSUM
    # Older ADNI exports use DXCURREN / DXCHANGE instead of DIAGNOSIS.
    if "DIAGNOSIS" not in dxsum.columns:
        dx_like = [c for c in dxsum.columns if str(c).upper().startswith("DX")]
        raise SystemExit(f"DXSUM has no DIAGNOSIS column. DX* columns present: {dx_like}")

    dxsum["PTID"] = dxsum["PTID"].astype(str).str.upper()
    dxsum["EXAMDATE_dt"] = pd.to_datetime(dxsum["EXAMDATE"], errors="coerce")

    # Take earliest visit with a non-null DIAGNOSIS as "baseline"
    dx_valid = dxsum.dropna(subset=["DIAGNOSIS"]).sort_values(["PTID", "EXAMDATE_dt"])
    dx_baseline = dx_valid.drop_duplicates(subset=["PTID"], keep="first").copy()
    dx_baseline["label"] = dx_baseline["DIAGNOSIS"].apply(map_diagnosis)

    dx_slim = dx_baseline[["PTID", "label", "EXAMDATE"]].rename(
        columns={"PTID": "subject_id", "EXAMDATE": "baseline_examdate"}
    )
    print(f"[DXSUM]   {len(dx_slim)} subjects with at least one diagnosed visit")
    print(f"[DXSUM]   {int(dx_slim['label'].notna().sum())} with a parseable label")

    # ----------------- Demographics from PTDEMOG
    ptdemog["PTID"] = ptdemog["PTID"].astype(str).str.upper()
    # Most subjects have one row; if multiple, pick the first
    demo = ptdemog.drop_duplicates(subset=["PTID"], keep="first").copy()

    # PTGENDER is text ("Male"/"Female") in some exports and numeric (1/2) in
    # others; handle both so sex does not silently come out empty.
    sex_raw = demo["PTGENDER"].astype(str).str.strip()
    sex_text = sex_raw.str[:1].str.upper()
    sex_numeric = sex_raw.str.replace(r"\.0$", "", regex=True).map({"1": "M", "2": "F"})
    demo["sex"] = sex_text.where(sex_text.isin(["M", "F"]), sex_numeric)
    demo["ptdob_ts"] = demo["PTDOB"].apply(parse_ptdob)

    demo_slim = demo[["PTID", "sex", "ptdob_ts"]].rename(columns={"PTID": "subject_id"})
    print(f"[PTDEMOG] {len(demo_slim)} unique subjects")
    print(f"[PTDEMOG] {int(demo_slim['sex'].notna().sum())} with parseable sex")
    print(f"[PTDEMOG] {int(demo_slim['ptdob_ts'].notna().sum())} with parseable PTDOB")

    # Merging
    raw["subject_id"] = raw["subject_id"].astype(str).str.upper()
    merged = raw.merge(dx_slim, on="subject_id", how="left")
    merged = merged.merge(demo_slim, on="subject_id", how="left")

    merged["age"] = merged.apply(
        lambda r: compute_age(r["ptdob_ts"], r["baseline_examdate"]),
        axis=1,
    )
    merged = merged.drop(columns=["ptdob_ts"])

    n_total = len(merged)
    n_with_label = int(merged["label"].notna().sum())
    n_with_age = int(merged["age"].notna().sum())
    print(f"\n[Merge] scans: {n_total}")
    print(f"[Merge] with label: {n_with_label}")
    print(f"[Merge] with age:   {n_with_age}")

    unmatched = merged[merged["label"].isna() & merged["age"].isna()]
    if len(unmatched) > 0:
        print(f"[Merge] {len(unmatched)} scans missing both label and age")
        print(f"        sample subject_ids: {unmatched['subject_id'].head(5).tolist()}")

    if not args.allow_missing_labels:
        merged = merged.dropna(subset=["label", "age"]).reset_index(drop=True)

    merged["label"] = merged["label"].astype("Int64")

    if len(merged) > 0:
        print()
        print(f"[Class balance] {merged['label'].value_counts(dropna=False).sort_index().to_dict()}")
        print(f"[Sex balance]   {merged['sex'].value_counts(dropna=False).to_dict()}")
        print(
            f"[Age summary]   mean={merged['age'].mean():.1f}  "
            f"range={merged['age'].min():.1f}-{merged['age'].max():.1f}"
        )
    else:
        print("[WARNING] zero rows survived label/age filtering")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nWrote in {args.out}")


if __name__ == "__main__":
    main()