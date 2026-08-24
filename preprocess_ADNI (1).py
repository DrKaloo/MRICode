"""
ADNI preprocessing pipeline matched bit-for-bit to OASIS preprocess_all.py.

The point of the match is so that cross-cohort generalisation (OASIS<->ADNI)
reflects cohort/scanner/acquisition differences, NOT preprocessing differences.

Directory structure expected:
    {INPUT_ROOT}/{subject_id}/{protocol}/{scan_datetime}/{image_id}/ADNI_*.nii

Dedup policy (one scan per subject):
    1. Group by (subject_id, series_id); keep Scaled_2 over Scaled
       (Scaled_2 = ADNI's latest reprocessing of the same source acquisition)
    2. Among remaining scans, prefer MPR over MPR-R (repeat acquisition),
       then earliest scan_date (baseline visit -- AD CNN literature convention)

Preprocessing (mirrors preprocess_all.py):
    1. Load NIfTI, squeeze 4D -> 3D
    2. 10th-percentile brain mask + morphological cleanup
       (binary_fill_holes, binary_erosion x1, binary_dilation x2)
    3. Within-mask 1st/99th percentile clipping
    4. Within-mask z-score normalisation; background -> 0
    5. Crop to brain bbox with 5-voxel margin (affine updated)
    6. Cubic resample to target^3 isotropic (affine updated)
    7. Atomic save via temp + os.replace
"""

import os
import re
import argparse
from glob import glob
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom, binary_fill_holes, binary_erosion, binary_dilation
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# Filename / path

# Example filename:
# ADNI_023_S_0388_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001152419019_S13076_I118853.nii
SUBJ_RE = re.compile(r"(\d{3}_S_\d{4})", re.IGNORECASE)
SERIES_RE = re.compile(r"_S(\d+)_I", re.IGNORECASE)
IMAGE_RE = re.compile(r"_I(\d+)\.nii", re.IGNORECASE)
DATE_DIR_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")  # e.g. 2006-04-10_09_53_55.0


def parse_scan(path: str) -> Optional[dict]:
    """Extract metadata from an ADNI .nii filepath. Returns None if unparseable."""
    fname = os.path.basename(path)
    parts = path.replace("\\", "/").split("/")

    # Subject id - prefer path component (more reliable than guessing filenames)
    subj = None
    for p in parts:
        m = SUBJ_RE.fullmatch(p)
        if m:
            subj = m.group(1).upper()
            break
    if subj is None:
        m = SUBJ_RE.search(fname)
        if m:
            subj = m.group(1).upper()
    if subj is None:
        return None

    m = SERIES_RE.search(fname)
    series = m.group(1) if m else None

    m = IMAGE_RE.search(fname)
    image_id = m.group(1) if m else None

    # Scan datetime from directory name (e.g. 2006-04-10_09_53_55.0)
    scan_date = None
    for p in parts:
        m = DATE_DIR_RE.match(p)
        if m:
            scan_date = m.group(1)
            break

    is_scaled_2 = ("_Scaled_2_" in fname) or ("_Scaled_2." in fname)
    is_repeat = "MPR-R" in fname

    return {
        "subject_id": subj,
        "scan_path_orig": path,
        "scan_date": scan_date,
        "series_id": series,
        "image_id": image_id,
        "is_scaled_2": int(is_scaled_2),
        "is_repeat": int(is_repeat),
    }


def dedupe_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Two-stages: series-level (prefer Scaled_2), then subject-level (baseline)."""
    # Stage 1: within (subject, series) keep Scaled_2 over Scaled
    df = df.sort_values(
        ["subject_id", "series_id", "is_scaled_2"],
        ascending=[True, True, False],
    )
    df = df.drop_duplicates(subset=["subject_id", "series_id"], keep="first")

    # Stage 2: within subject prefer MPR over MPR-R, then earliest scan_date
    df = df.sort_values(
        ["subject_id", "is_repeat", "scan_date"],
        ascending=[True, True, True],
        na_position="last",
    )
    df = df.drop_duplicates(subset=["subject_id"], keep="first")

    return df.reset_index(drop=True)


# Preprocessing (keeping it the same as preprocess_all.py)

def atomic_save(img: nib.Nifti1Image, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path.replace(".nii.gz", ".tmp.nii.gz")
    nib.save(img, tmp)
    nib.load(tmp)
    os.replace(tmp, out_path)


def preprocess_volume(
    input_path: str,
    output_path: str,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
) -> bool:
    img = nib.load(input_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine.astype(np.float64).copy()

    if data.ndim == 4:
        data = np.squeeze(data)
    if data.ndim != 3:
        return False

    # 10th-percentile brain mask of nonzero voxels (matches OASIS pipeline)
    nonzero = data[data > 0]
    if nonzero.size < 1000: #1000
        return False
    thresh = np.percentile(nonzero, 10)

    mask = data > thresh
    mask = binary_fill_holes(mask)
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=2)
    if mask.sum() < 1000:
        return False

    # Intensity clipping within mask
    brain = data[mask]
    p1, p99 = np.percentile(brain, [1, 99])
    data = np.clip(data, p1, p99)

    # Z-score within mask, zero background
    brain_clipped = data[mask]
    mu = float(brain_clipped.mean())
    sigma = float(brain_clipped.std()) + 1e-8
    data = (data - mu) / sigma
    data[~mask] = 0.0

    # Crop to brain bbox + 5-voxel margin
    coords = np.array(np.where(mask))
    x0, y0, z0 = coords.min(axis=1)
    x1, y1, z1 = coords.max(axis=1) + 1
    pad = 5
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad); z0 = max(0, z0 - pad)
    x1 = min(data.shape[0], x1 + pad)
    y1 = min(data.shape[1], y1 + pad)
    z1 = min(data.shape[2], z1 + pad)

    cropped = data[x0:x1, y0:y1, z0:z1]
    if min(cropped.shape) < 8:
        return False

    # Update affine for crop translation: new origin is old affine applied to (x0,y0,z0)
    affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.array([x0, y0, z0], dtype=np.float64)

    # Resample to target shape (cubic) and update affine for zoom
    zooms = [target_shape[i] / cropped.shape[i] for i in range(3)]
    resampled = zoom(cropped, zooms, order=3).astype(np.float32)
    affine[:3, :3] = affine[:3, :3] @ np.diag([1.0 / z for z in zooms])

    out_img = nib.Nifti1Image(resampled, affine.astype(np.float32))
    atomic_save(out_img, output_path)
    return True


# Main Part

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_root",
        default=r"D:\ADNI\ADNI",
        help="Root containing per-subject folders (e.g. 023_S_0388/).",
    )
    ap.add_argument(
        "--out_dir",
        default=str(DATA_DIR / "processed_adni_128"),
        help="Where to write preprocessed .nii.gz files (one per subject).",
    )
    ap.add_argument(
        "--metadata_csv",
        default=str(DATA_DIR / "adni_raw_metadata.csv"),
        help="Where to write raw metadata CSV (pre-label-merge).",
    )
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process this many subjects (for dry-run testing).",
    )
    args = ap.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"input_root not found: {input_root}")

    print(f"[Scan] walking {input_root} ...")
    all_niftis = sorted(glob(str(input_root / "**" / "*.nii"), recursive=True))
    print(f"[Scan] found {len(all_niftis)} candidate .nii files")

    parsed = []
    for p in all_niftis:
        rec = parse_scan(p)
        if rec is not None:
            parsed.append(rec)
    print(f"[Parse] parseable: {len(parsed)}/{len(all_niftis)}")

    if not parsed:
        raise SystemExit("No parseable scans found.")

    df = pd.DataFrame(parsed)

    n_pre = len(df)
    df = dedupe_subjects(df)
    n_subjects = df["subject_id"].nunique()
    print(f"{n_pre} candidate scans -> {len(df)} chosen ({n_subjects} unique subjects)")

    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[Limit] processing first {len(df)} for dry run")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_shape = (args.resolution,) * 3

    success_paths = []
    fail_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ADNI preprocess"):
        subj = row["subject_id"]
        out_path = out_dir / f"{subj}.nii.gz"
        try:
            ok = preprocess_volume(row["scan_path_orig"], str(out_path), target_shape)
        except Exception as e:
            print(f"[Fail] {subj}: {e}")
            ok = False
        if ok:
            success_paths.append(str(out_path))
        else:
            success_paths.append(None)
            fail_count += 1

    df["scan_path"] = success_paths
    df["resolution"] = args.resolution
    df["dataset"] = "ADNI"
    df = df.dropna(subset=["scan_path"]).reset_index(drop=True)

    cols = [
        "dataset", "subject_id", "scan_path", "scan_date", "image_id",
        "series_id", "is_scaled_2", "is_repeat", "scan_path_orig", "resolution",
    ]
    df = df[[c for c in cols if c in df.columns]]

    Path(args.metadata_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.metadata_csv, index=False)

    print(f"Success={len(df)} fail={fail_count}")
    print(f"[Meta] written in {args.metadata_csv}")
    print(f"[Out]  written in {len(df)} preprocessed scans to {out_dir}")


if __name__ == "__main__":
    main()
