import os
import re
from glob import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom, binary_fill_holes
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

NIFTI_ROOT = r"C:\Users\todor\Desktop\ADNI DATABASE\ADNI_NIFTI"
RESOLUTION = 128



# Atomic save

def atomic_nifti_save(img: nib.Nifti1Image, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path.replace(".nii.gz", ".tmp.nii.gz")
    nib.save(img, tmp)
    nib.load(tmp)
    os.replace(tmp, out_path)



# ADNI-safe preprocessing

def preprocess_scan(
    input_path: str,
    output_path: str,
    target_shape: Tuple[int, int, int],
) -> bool:
    img = nib.load(input_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine

    if data.ndim == 4:
        data = np.squeeze(data)
    if data.ndim != 3:
        return False

    # Robust brain mask (NO data > 0 assumption)
    abs_data = np.abs(data)
    thresh = np.percentile(abs_data, 20)
    mask = abs_data > thresh
    mask = binary_fill_holes(mask)

    if mask.sum() < 1000:
        return False

    brain = data[mask]
    p1, p99 = np.percentile(brain, [1, 99])
    data = np.clip(data, p1, p99)

    mean = brain.mean()
    std = brain.std() + 1e-8
    data = (data - mean) / std
    data[~mask] = 0.0

    coords = np.array(np.where(mask))
    x0, y0, z0 = coords.min(axis=1)
    x1, y1, z1 = coords.max(axis=1)

    pad = 5
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    z0 = max(0, z0 - pad)
    x1 = min(data.shape[0], x1 + pad)
    y1 = min(data.shape[1], y1 + pad)
    z1 = min(data.shape[2], z1 + pad)

    cropped = data[x0:x1, y0:y1, z0:z1]
    if min(cropped.shape) < 8:
        return False

    zoom_factors = [target_shape[i] / cropped.shape[i] for i in range(3)]
    resized = zoom(cropped, zoom_factors, order=3).astype(np.float32)

    out_img = nib.Nifti1Image(resized, affine)
    atomic_nifti_save(out_img, output_path)
    return True



# Subject ID inference

SUBJ_RE = re.compile(r"(\d{3}_S_\d{4})", re.IGNORECASE)

def infer_subject_id(path: str) -> str | None:
    m = SUBJ_RE.search(path)
    return m.group(1).upper() if m else None



# Main

def main():
    out_dir = DATA_DIR / f"processed_adni_{RESOLUTION}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    success = 0
    fail = 0

    niftis = sorted(glob(os.path.join(NIFTI_ROOT, "*.nii*")))

    for p in tqdm(niftis, desc="ADNI preprocess"):
        sid = infer_subject_id(p)
        if sid is None:
            continue

        out_path = out_dir / f"{sid}.nii.gz"

        ok = preprocess_scan(p, str(out_path), (RESOLUTION,)*3)
        if not ok:
            fail += 1
            continue

        success += 1
        rows.append({
            "dataset": "ADNI",
            "subject_id": sid,
            "scan_path": str(out_path),
            "resolution": RESOLUTION,
        })

    df = pd.DataFrame(rows)
    meta_path = DATA_DIR / "adni_metadata.csv"
    df.to_csv(meta_path, index=False)

    print(f"[Done] success={success} fail={fail}")
    print(f"[Meta] {meta_path}")


if __name__ == "__main__":
    main()
