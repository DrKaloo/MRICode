import os
import re
import argparse
from glob import glob

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom, binary_erosion, binary_dilation, binary_fill_holes
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_FOLDER = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_BASE = os.path.join(BASE_DIR, "data")

# OAS2 demographics Csv:
# Msc-AD/data/oasis_longitudinal_demographics.csv (local dir)
OAS2_DEMOGRAPHICS_PATH = os.path.join(OUTPUT_BASE, "oasis_longitudinal_demographics.csv")

AGE_MIN = None
AGE_MAX = None
SEX_FILTER = None  # 'M', 'F', or None
EDUC_MIN = None

RESOLUTIONS = [96, 128]

# Default Behaviour:
# wipes processed_{res} outputs
# re-preprocess all scans
# regenerates metadata.csv
DEFAULT_WIPE = True
DEFAULT_PREPROCESS = True



# Atomic save

def atomic_nifti_save(img: nib.Nifti1Image, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if out_path.endswith(".nii.gz"):
        tmp_path = out_path[:-7] + ".tmp.nii.gz"
    elif out_path.endswith(".nii"):
        tmp_path = out_path[:-4] + ".tmp.nii"
    else:
        tmp_path = out_path + ".tmp.nii.gz"

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    nib.save(img, tmp_path)

    if (not os.path.exists(tmp_path)) or os.path.getsize(tmp_path) == 0:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(f"Atomic save has failed (empty tmp): {tmp_path}")

    _ = nib.load(tmp_path)
    os.replace(tmp_path, out_path)

    if os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Atomic save failed (empty out): {out_path}")



# Preprocess one scan

def preprocess_scan_proper(input_path: str, output_path: str, target_shape: tuple[int, int, int]):
    img = nib.load(input_path)
    data = img.get_fdata(dtype=np.float32)
    original_affine = img.affine

    if data.ndim == 4:
        data = np.squeeze(data)
    if data.ndim != 3:
        return None

    pos = data[data > 0]
    if pos.size < 100:
        return None

    threshold = np.percentile(pos, 10)
    brain_mask = data > threshold

    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = binary_erosion(brain_mask, iterations=1)
    brain_mask = binary_dilation(brain_mask, iterations=2)

    if not np.any(brain_mask):
        return None

    brain_voxels = data[brain_mask]
    if brain_voxels.size < 100:
        return None

    p01, p99 = np.percentile(brain_voxels, [1, 99])
    data_clipped = np.clip(data, p01, p99)

    brain_voxels_clipped = data_clipped[brain_mask]
    mean_val = float(brain_voxels_clipped.mean())
    std_val = float(brain_voxels_clipped.std())

    data_norm = (data_clipped - mean_val) / (std_val + 1e-8)
    data_norm[~brain_mask] = 0.0

    coords = np.array(np.where(brain_mask))
    if coords.size == 0:
        return None

    x_min, y_min, z_min = coords.min(axis=1)
    x_max, y_max, z_max = coords.max(axis=1)

    pad = 5
    x_min = max(0, int(x_min - pad))
    y_min = max(0, int(y_min - pad))
    z_min = max(0, int(z_min - pad))
    x_max = min(data.shape[0], int(x_max + pad + 1))
    y_max = min(data.shape[1], int(y_max + pad + 1))
    z_max = min(data.shape[2], int(z_max + pad + 1))

    cropped = data_norm[x_min:x_max, y_min:y_max, z_min:z_max]
    if min(cropped.shape) < 4:
        return None

    zoom_factors = [target_shape[i] / cropped.shape[i] for i in range(3)]
    resized = zoom(cropped, zoom_factors, order=3).astype(np.float32, copy=False)

    crop_translation = np.array([x_min, y_min, z_min], dtype=float)
    new_affine = original_affine.copy()
    new_affine[:3, 3] += original_affine[:3, :3] @ crop_translation

    zoom_matrix = np.diag([1 / zoom_factors[0], 1 / zoom_factors[1], 1 / zoom_factors[2], 1.0])
    new_affine = new_affine @ zoom_matrix

    out_img = nib.Nifti1Image(resized, affine=new_affine)
    atomic_nifti_save(out_img, output_path)
    return target_shape



# Discover scans (OAS1 + OAS2)

def _discover_oas1_scans(base_folder: str) -> list[str]:
    session_folders = glob(os.path.join(base_folder, "OAS1_*_MR*"))
    scan_files: list[str] = []

    for session_folder in session_folders:
        processed_folder = os.path.join(session_folder, "Processed", "MPRAGE", "T88_111")
        if os.path.exists(processed_folder):
            scans = glob(os.path.join(processed_folder, "*_masked_gfc.hdr"))
            if not scans:
                scans = glob(os.path.join(processed_folder, "*_gfc.hdr"))
            if scans:
                scan_files.extend(scans)
                continue

        raw_folder = os.path.join(session_folder, "Raw")
        if os.path.exists(raw_folder):
            scans = glob(os.path.join(raw_folder, "*_anon.hdr"))
            scan_files.extend(scans)

    return scan_files


def _discover_oas2_scans(base_folder: str) -> list[str]:
    session_folders = glob(os.path.join(base_folder, "**", "OAS2_*_MR*"), recursive=True)
    scan_files: list[str] = []

    for session_folder in session_folders:
        raw_folder = os.path.join(session_folder, "Raw")
        if not os.path.exists(raw_folder):
            continue

        scans = glob(os.path.join(raw_folder, "mpr-*.nifti.hdr"))
        if not scans:
            scans = glob(os.path.join(raw_folder, "*.hdr"))

        scan_files.extend(scans)

    return scan_files


def find_oasis_scans() -> list[str]:
    scan_files = []
    scan_files.extend(_discover_oas1_scans(BASE_FOLDER))
    scan_files.extend(_discover_oas2_scans(BASE_FOLDER))

    seen = set()
    out = []
    for p in scan_files:
        n = os.path.normpath(p)
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out



# Output management + Storage

def wipe_output_folders():
    for res in RESOLUTIONS:
        folder = os.path.join(OUTPUT_BASE, f"processed_{res}")
        if not os.path.exists(folder):
            continue

        for f in glob(os.path.join(folder, "*.nii.gz")):
            try:
                os.remove(f)
            except OSError:
                pass

        for f in glob(os.path.join(folder, "*.tmp.nii*")):
            try:
                os.remove(f)
            except OSError:
                pass


def _session_id_from_path(scan_path: str) -> str | None:
    parts = scan_path.split(os.sep)
    for part in parts:
        if part.startswith("OAS1_") and "_MR" in part:
            return part
        if part.startswith("OAS2_") and "_MR" in part:
            return part
    return None


def _subject_id_from_session(session_id: str) -> str:
    m = re.search(r"((?:OAS1|OAS2)_\d{4})", str(session_id), re.IGNORECASE)
    return m.group(1) if m else str(session_id)


def process_all_resolutions(force: bool = True):
    scan_files = find_oasis_scans()
    n_oas1 = sum(1 for p in scan_files if "OAS1_" in p)
    n_oas2 = sum(1 for p in scan_files if "OAS2_" in p)
    print(f"Found {len(scan_files)} scans (OAS1+OAS2)")
    print(f"To Debug scan breakdown: OAS1={n_oas1} OAS2={n_oas2}")

    for res in RESOLUTIONS:
        out_folder = os.path.join(OUTPUT_BASE, f"processed_{res}")
        os.makedirs(out_folder, exist_ok=True)

        success = 0
        fail = 0

        for scan_path in tqdm(scan_files, desc=f"Processing {res}^3"):
            session_id = _session_id_from_path(scan_path)
            if session_id is None:
                continue

            filename = os.path.basename(scan_path).replace(".hdr", "")
            output_path = os.path.join(out_folder, f"{session_id}_{filename}.nii.gz")

            if output_path.endswith(".nii.gz"):
                tmp_path = output_path[:-7] + ".tmp.nii.gz"
            else:
                tmp_path = output_path + ".tmp.nii.gz"

            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                try:
                    os.remove(output_path)
                except OSError:
                    pass

            # If not forcing, ignore existing outputs
            if (not force) and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                continue

            # Delete existing output to re- run
            if force and os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

            try:
                ok = preprocess_scan_proper(scan_path, output_path, (res, res, res))
                if ok is not None:
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                try:
                    if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                        os.remove(output_path)
                except OSError:
                    pass
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                print(f"\nFail {session_id} ({res}): {type(e).__name__}: {e}")

        print(f"Completed {res}^3: success={success} fail={fail}")



# Demographics + labels

def _cdr_to_label(cdr: float) -> tuple[str, int] | None:
    if cdr == 0:
        return "CN", 0
    if cdr == 0.5:
        return "VeryMild", 1
    if cdr >= 1.0:
        return "Dementia", 2
    return None


def _load_oas2_table() -> pd.DataFrame | None:
    if os.path.exists(OAS2_DEMOGRAPHICS_PATH):
        f = OAS2_DEMOGRAPHICS_PATH
        print(f"OAS2 demographics file: {f}")
    else:
        patterns = [
            os.path.join(OUTPUT_BASE, "**", "*demographic*.csv"),
            os.path.join(OUTPUT_BASE, "**", "*longitudinal*.csv"),
            os.path.join(BASE_FOLDER, "**", "*demographic*.csv"),
            os.path.join(BASE_FOLDER, "**", "*longitudinal*.csv"),
        ]
        cands = []
        for pat in patterns:
            cands.extend(glob(pat, recursive=True))
        cands = [c for c in cands if os.path.isfile(c)]
        cands.sort(key=lambda p: (len(p), p.lower()))
        f = cands[0] if cands else None
        print(f"OAS2 demographics candidate: {f}")

    if not f:
        return None

    ext = os.path.splitext(f)[1].lower()
    if ext != ".csv":
        raise RuntimeError(f"OAS2 demographics must be CSV! Found: {f}")

    read_errors = []
    df = None
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(f, sep=None, engine="python", encoding=enc)
            break
        except Exception as e:
            read_errors.append(f"{enc}:{type(e).__name__}")
            df = None
    if df is None:
        raise RuntimeError(f"Failed to read OAS2 .csv ({f}). Attempts: {read_errors}")

    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    c_session = pick("mri_id", "mriid", "mri", "scan_id", "scanid", "session_id", "sessionid", "image_id", "imageid")
    c_subj = pick("subject_id", "subjectid", "subject", "id", "participant_id")
    c_cdr = pick("cdr", "cdr_global", "cdr_total", "cdr_score")
    c_age = pick("age", "age_at_visit", "age_at_scan", "age_at_mri")
    c_sex = pick("sex", "gender", "m_f", "m/f")
    c_educ = pick("educ", "education", "years_education", "education_years")
    c_mmse = pick("mmse", "mini_mental_state_exam", "minimental")

    if c_session is None:
        print(f"Warning: OAS2 table missing a session/MRI column. Columns={df.columns.tolist()}")
        return None

    out = pd.DataFrame()
    out["session_id"] = df[c_session].astype(str).str.strip()

    if c_subj is not None:
        out["subject_id"] = df[c_subj].astype(str).str.strip()
    else:
        out["subject_id"] = out["session_id"].str.extract(r"(OAS2_\d{4})", expand=False)

    out["cdr"] = pd.to_numeric(df[c_cdr], errors="coerce") if c_cdr else np.nan
    out["age"] = pd.to_numeric(df[c_age], errors="coerce") if c_age else np.nan
    out["sex"] = df[c_sex].astype(str).str.strip() if c_sex else ""
    out["educ"] = pd.to_numeric(df[c_educ], errors="coerce") if c_educ else np.nan
    out["mmse"] = pd.to_numeric(df[c_mmse], errors="coerce") if c_mmse else np.nan

    out["sex"] = (
        out["sex"]
        .replace({"male": "M", "female": "F", "m": "M", "f": "F", "1": "M", "0": "F"})
        .astype(str)
        .str.upper()
        .str.strip()
    )
    out.loc[~out["sex"].isin(["M", "F"]), "sex"] = ""

    out = out[out["session_id"].str.contains(r"^OAS2_\d{4}_MR\d+", case=False, regex=True)].copy()

    out["_cdr_ok"] = out["cdr"].notna().astype(int)
    out = (
        out.sort_values(["session_id", "_cdr_ok"], ascending=[True, False])
           .drop_duplicates(subset=["session_id"], keep="first")
           .drop(columns=["_cdr_ok"])
    )

    print(f"OAS2 table rows usable: {len(out)} (unique sessions)")
    return out


def extract_labels_with_demographics():
    rows = []

    # OAS1: per-session .txt files
    oas1_sessions = glob(os.path.join(BASE_FOLDER, "OAS1_*_MR*"))
    skipped_oas1_no_cdr = 0

    for session_folder in oas1_sessions:
        session_id = os.path.basename(session_folder)
        subject_id = _subject_id_from_session(session_id)

        txt_file = os.path.join(session_folder, f"{session_id}.txt")
        if not os.path.exists(txt_file):
            continue

        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        cdr = age = educ = mmse = None
        sex = None

        for line in content.splitlines():
            line = line.strip()
            if "CDR:" in line:
                try:
                    cdr = float(line.split("CDR:")[1].strip())
                except Exception:
                    pass
            elif "AGE:" in line:
                try:
                    age = int(line.split("AGE:")[1].strip())
                except Exception:
                    pass
            elif "M/F:" in line:
                try:
                    sex = line.split("M/F:")[1].strip()
                except Exception:
                    pass
            elif "EDUC:" in line:
                try:
                    educ = int(line.split("EDUC:")[1].strip())
                except Exception:
                    pass
            elif "MMSE:" in line:
                try:
                    mmse = int(line.split("MMSE:")[1].strip())
                except Exception:
                    pass

        if cdr is None:
            skipped_oas1_no_cdr += 1
            continue

        if AGE_MIN is not None and (age is None or age < AGE_MIN):
            continue
        if AGE_MAX is not None and (age is None or age > AGE_MAX):
            continue
        if SEX_FILTER is not None and sex != SEX_FILTER:
            continue
        if EDUC_MIN is not None and (educ is None or educ < EDUC_MIN):
            continue

        diag = _cdr_to_label(cdr)
        if diag is None:
            continue
        diagnosis, label = diag

        for res in RESOLUTIONS:
            proc_folder = os.path.join(OUTPUT_BASE, f"processed_{res}")
            pattern = os.path.join(proc_folder, f"{session_id}_*.nii.gz")
            for scan_path in glob(pattern):
                if (not os.path.exists(scan_path)) or os.path.getsize(scan_path) == 0:
                    continue
                rows.append({
                    "dataset": "OAS1",
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "resolution": res,
                    "scan_filename": os.path.basename(scan_path),
                    "scan_path": scan_path,
                    "cdr": cdr,
                    "age": age,
                    "sex": sex,
                    "educ": educ,
                    "mmse": mmse,
                    "diagnosis": diagnosis,
                    "label": label,
                })

    # OAS2: demographics table matched by session_id (unique p)
    oas2_table = _load_oas2_table()
    oas2_sessions = glob(os.path.join(BASE_FOLDER, "**", "OAS2_*_MR*"), recursive=True)

    if oas2_table is None or len(oas2_table) == 0:
        print("Warning: No usable OAS2 demographics table found. Consequently OAS2 preprocessed but not added to metadata.csv.")
    else:
        t = oas2_table.copy()
        t["session_id"] = t["session_id"].astype(str).str.strip()
        t_idx = {sid: row for sid, row in t.set_index("session_id").iterrows()}

        skipped_oas2_no_match = 0
        skipped_oas2_no_cdr = 0
        added_oas2 = 0

        for session_folder in oas2_sessions:
            session_id = os.path.basename(session_folder)
            subject_id = _subject_id_from_session(session_id)

            r = t_idx.get(session_id)
            if r is None:
                skipped_oas2_no_match += 1
                continue

            cdr = float(r["cdr"]) if pd.notna(r.get("cdr", np.nan)) else None
            if cdr is None:
                skipped_oas2_no_cdr += 1
                continue

            age = None if pd.isna(r.get("age", np.nan)) else float(r["age"])
            sex = str(r.get("sex", "")).strip() if isinstance(r.get("sex", ""), str) else ""
            sex = sex if sex in ["M", "F"] else None
            educ = None if pd.isna(r.get("educ", np.nan)) else int(r["educ"])
            mmse = None if pd.isna(r.get("mmse", np.nan)) else int(r["mmse"])

            if AGE_MIN is not None and (age is None or age < AGE_MIN):
                continue
            if AGE_MAX is not None and (age is None or age > AGE_MAX):
                continue
            if SEX_FILTER is not None and sex != SEX_FILTER:
                continue
            if EDUC_MIN is not None and (educ is None or educ < EDUC_MIN):
                continue

            diag = _cdr_to_label(cdr)
            if diag is None:
                continue
            diagnosis, label = diag

            for res in RESOLUTIONS:
                proc_folder = os.path.join(OUTPUT_BASE, f"processed_{res}")
                pattern = os.path.join(proc_folder, f"{session_id}_*.nii.gz")
                for scan_path in glob(pattern):
                    if (not os.path.exists(scan_path)) or os.path.getsize(scan_path) == 0:
                        continue
                    rows.append({
                        "dataset": "OAS2",
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "resolution": res,
                        "scan_filename": os.path.basename(scan_path),
                        "scan_path": scan_path,
                        "cdr": cdr,
                        "age": age,
                        "sex": sex,
                        "educ": educ,
                        "mmse": mmse,
                        "diagnosis": diagnosis,
                        "label": label,
                    })
                    added_oas2 += 1

        print(f"OAS2 added_rows={added_oas2} skipped_no_match={skipped_oas2_no_match} skipped_no_cdr={skipped_oas2_no_cdr}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_BASE, "metadata.csv")
    df.to_csv(out_csv, index=False)

    print(f"Wrote: {out_csv}")
    if len(df):
        print(f"Scan entries: {len(df)} | Unique subjects: {df['subject_id'].nunique()}")
        print(df["dataset"].value_counts().to_string())
    print(f"OAS1 skipped_no_cdr={skipped_oas1_no_cdr}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_wipe", action="store_true", help=             "Do not delete existing processed_*/ volumes first")
    ap.add_argument("--skip_preprocess", action="store_true", help=     "Skip volume preprocessing; only write metadata.csv")
    ap.add_argument("--no_force", action="store_true", help=            "If preprocessing, do not overwrite existing outputs")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()

    print("OASIS Preprocessing (OAS1 + OAS2; atomic saves)")

    do_wipe = DEFAULT_WIPE and (not args.no_wipe)
    do_preprocess = DEFAULT_PREPROCESS and (not args.skip_preprocess)
    force = not args.no_force

    if do_preprocess:
        if do_wipe:
            print("Run wipe_output_folders()")
            wipe_output_folders()
        print(f"Run process_all_resolutions(force={force})")
        process_all_resolutions(force=force)
    else:
        print("[Skip] process_all_resolutions()")

    print("Run extract_labels_with_demographics()")
    extract_labels_with_demographics()