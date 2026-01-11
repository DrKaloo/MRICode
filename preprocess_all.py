import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom, binary_erosion, binary_dilation, binary_fill_holes
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

"""
OASIS Dataset Preprocessing Pipeline

Complete data preprocessing pipeline for OASIS brain MRI dataset, including
skull stripping, intensity normalization, resampling, metadata extraction,
and train/validation/test split generation.

Processing Steps:
    1. Scan preprocessing (skull stripping, normalization, resampling)
    2. Metadata extraction (CDR, age, sex, education, MMSE)
    3. Label assignment (CDR 0 → CN, 0.5 → VeryMild, ≥1 → Dementia)
    4. Train/val/test split creation (stratified, reproducible)

Preprocessing Pipeline per Scan:
    1. Load NIfTI file (.nii or .hdr format)
    2. Brain mask creation (percentile threshold + morphological ops)
    3. Intensity normalization (clip outliers + Z-score)
    4. Crop to brain bounding box (minimal volume)
    5. High-quality resampling (cubic interpolation)
    6. Affine matrix adjustment (preserve spatial coordinates)
    7. Save compressed NIfTI (.nii.gz)

Data Organization:
    Input:  data/raw/OAS1_*_MR*/...
    Output: data/processed_96/    (96³ resolution scans)
            data/processed_128/   (128³ resolution scans)
            data/metadata.csv     (demographics + labels)
            data/splits/          (train/val/test CSVs)

Train/Val/Test Split:
    - Split ratio: 60% / 20% / 20%
    - Stratified by class (balanced splits)
    - One scan per patient (prevents data leakage)
    - Random seed: 42 (reproducible)
    - Separate splits for binary and 3-class tasks

Binary Task (Early Detection):
    - CN (CDR=0) vs VeryMild (CDR=0.5)
    - ~205 patients total
    - Used for main experiments

3-Class Task (Disease Staging):
    - CN vs VeryMild vs Dementia (CDR≥1)
    - ~416 patients total
    - Not used in this project

Demographic Filters (Currently Disabled):
    - AGE_MIN / AGE_MAX: Age range filter
    - SEX_FILTER: 'M', 'F', or None
    - EDUC_MIN: Minimum education years
    All set to None = use all available data

Usage:
    python preprocess_all.py

Prerequisites:
    - OASIS dataset downloaded to data/raw/
    - Required packages: nibabel, scipy, sklearn, pandas, numpy, tqdm

Outputs:
    - Preprocessed scans in data/processed_*/
    - metadata.csv with demographics
    - Train/val/test splits in data/splits/
"""

BASE_FOLDER = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\raw"
OUTPUT_BASE = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data"

#demographics
#noinspection PyTypeChecker
AGE_MIN = None
AGE_MAX = None
SEX_FILTER = None  #'M', 'F', or None
EDUC_MIN = None


def preprocess_scan_proper(input_path, output_path, target_shape):
    """
        Preprocess a single MRI scan with skull stripping and normalization.

        Complete preprocessing pipeline:
            1. Load NIfTI file and preserve spatial information (affine matrix)
            2. Create brain mask using percentile threshold
            3. Morphological cleanup (fill holes, erosion, dilation)
            4. Intensity normalization (percentile clipping + Z-score)
            5. Crop to brain bounding box with padding
            6. High-quality resampling to target resolution (cubic interpolation)
            7. Adjust affine matrix for cropping and resampling
            8. Save as compressed NIfTI

        Args:
            input_path (str): Path to input NIfTI file (.nii, .hdr, or .nii.gz)
            output_path (str): Path for output .nii.gz file
            target_shape (tuple): Target 3D shape, e.g., (128, 128, 128)

        Returns:
            tuple or None: Target shape if successful, None if failed

        Processing Details:
            - Brain mask: 10th percentile threshold (robust to noise)
            - Normalization: Clip to [1st, 99th] percentile, then Z-score
            - Resampling: Cubic interpolation (order=3) for high quality
            - Padding: 5 voxels around brain bounding box
            - Affine: Adjusted to preserve anatomical coordinates

        Example:
            result = preprocess_scan_proper(
                'raw/scan.nii',
                'processed/scan.nii.gz',
                (128, 128, 128)
            )
        """

    img = nib.load(input_path)
    data = img.get_fdata() # type: ignore
    original_affine = img.affine # type: ignore
    #(line above) keeping original spatial information/geometry - preserving spatial information

    if data.ndim == 4:
        data = data.squeeze()

    #Better brain mask using percentile-based threshold
    threshold = np.percentile(data[data > 0], 10)
    brain_mask = data > threshold

    #Morphological cleanup
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = binary_erosion(brain_mask, iterations=1)
    brain_mask = binary_dilation(brain_mask, iterations=2)

    #Intensity normalisation
    brain_voxels = data[brain_mask]
    if len(brain_voxels) < 100:
        return None

    #Use percentile normalisation (more robust than mean/std)
    p01, p99 = np.percentile(brain_voxels, [1, 99])
    data_clipped = np.clip(data, p01, p99)

#noinspection DuplicatedCode
    #Z-score normalisation
    mean_val = brain_voxels.mean()
    std_val = brain_voxels.std()
    data_normalized = (data_clipped - mean_val) / (std_val + 1e-8)
    data_normalized[~brain_mask] = 0

    #Crop image to brain
    # noinspection DuplicatedCode
    coords = np.array(np.where(brain_mask))
    x_min, y_min, z_min = coords.min(axis=1)
    x_max, y_max, z_max = coords.max(axis=1)

    pad = 5
    x_min = max(0, x_min - pad)
    x_max = min(data.shape[0], x_max + pad + 1)  # +1 to include endpoint
    y_min = max(0, y_min - pad)
    y_max = min(data.shape[1], y_max + pad + 1)
    z_min = max(0, z_min - pad)
    z_max = min(data.shape[2], z_max + pad + 1)

    data_cropped = data_normalized[x_min:x_max, y_min:y_max, z_min:z_max]

    #High quality resampling
    zoom_factors = [target_shape[i] / data_cropped.shape[i] for i in range(3)]
    data_resized = zoom(data_cropped, zoom_factors, order=3)  #Cubic interpolation

    #Create proper affine for resised image
    #Adjust affine to account for cropping and resampling
    crop_translation = np.array([x_min, y_min, z_min])
    new_affine = original_affine.copy()
    new_affine[:3, 3] += original_affine[:3, :3] @ crop_translation

    #Adjust for resampling
    zoom_matrix = np.diag([1/zoom_factors[0], 1/zoom_factors[1], 1/zoom_factors[2], 1])
    new_affine = new_affine @ zoom_matrix

    #Save with proper affine
    output_img = nib.Nifti1Image(data_resized, affine=new_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(output_img, output_path)

    return target_shape



def find_processed_scans():
    """
        Locate all MRI scans in OASIS dataset folder structure.

        Searches for scans with priority:
            1. Processed scans: PROCESSED/MPRAGE/T88_111/*.hdr (pre-aligned, skull-stripped)
            2. Raw scans: RAW/*_anon.hdr (if processed not available)

        OASIS folder structure:
            data/raw/
                OAS1_0001_MR1/
                    PROCESSED/MPRAGE/T88_111/*.hdr  ← Priority 1
                    RAW/*_anon.hdr                   ← Priority 2
                    OAS1_0001_MR1.txt                ← Metadata
                OAS1_0002_MR1/
                    ...

        Returns:
            list: Paths to all found scan files (.hdr format)
        """

    patient_folders = glob(os.path.join(BASE_FOLDER, "OAS1_*_MR*"))
    scan_files = []

    for patient_folder in patient_folders:
        #better quality first once processed
        #noinspection SpellCheckingInspection
        processed_folder = os.path.join(patient_folder, "PROCESSED", "MPRAGE", "T88_111")
        if os.path.exists(processed_folder):
            scans = glob(os.path.join(processed_folder, "*.hdr"))
            if scans:
                scan_files.extend(scans)
                continue

        #Raw 1st if None In Processed Path
        raw_folder = os.path.join(patient_folder, "RAW")
        if os.path.exists(raw_folder):
            scans = glob(os.path.join(raw_folder, "*_anon.hdr"))
            scan_files.extend(scans)

    return scan_files


def process_all_resolutions():
    """
        Process all scans at multiple resolutions (96³ and 128³).

        For each resolution:
            1. Find all available scans
            2. Extract patient ID from path
            3. Process scan (skull strip, normalize, resample)
            4. Save to data/processed_{resolution}/
            5. Track success/failure rates

        Outputs:
            - data/processed_96/{patient_id}_{filename}.nii.gz
            - data/processed_128/{patient_id}_{filename}.nii.gz

        Progress:
            Shows tqdm progress bar for each resolution
            Reports success rate at completion
        """

    scan_files = find_processed_scans()

    print(f"Found {len(scan_files)} scans\n")

    for res in [96, 128]:
        print(f"Processing {res}³...")
        output_folder = os.path.join(OUTPUT_BASE, f"processed_{res}")

        success = 0
        for scan_path in tqdm(scan_files, desc=f"Resolution {res}³"):
            parts = scan_path.split(os.sep)

            #Extract patient ID
            patient_id = None
            for part in parts:
                if part.startswith('OAS1_'):
                    patient_id = part
                    break

            if patient_id:
                filename = os.path.basename(scan_path).replace('.hdr', '')
                output_path = os.path.join(output_folder, f"{patient_id}_{filename}.nii.gz")

                try:
                    result = preprocess_scan_proper(scan_path, output_path, (res, res, res))
                    if result:
                        success += 1
                except (IOError, ValueError, RuntimeError) as e:
                    print(f"\nError on {patient_id}: {e}")

        print(f"Completed {res}³: {success}/{len(scan_files)} successful")


def extract_labels_with_demographics():
    """
        Extract metadata, labels, and demographics from OASIS text files.

        For each patient:
            1. Read patient-specific .txt file (e.g., OAS1_0001_MR1.txt)
            2. Parse metadata: CDR, age, sex, education, MMSE
            3. Apply demographic filters (if enabled)
            4. Assign labels: CDR 0→CN, 0.5→VeryMild, ≥1→Dementia
            5. Match to processed scans
            6. Save to metadata.csv

        Label Mapping:
            - CDR = 0   → Label 0 (CN - Cognitively Normal)
            - CDR = 0.5 → Label 1 (VeryMild - Very Mild Impairment)
            - CDR ≥ 1.0 → Label 2 (Dementia - Mild+ Dementia)

        Demographic Filters (configurable at top of file):
            - AGE_MIN / AGE_MAX: Age range
            - SEX_FILTER: 'M', 'F', or None
            - EDUC_MIN: Minimum education years
            Currently all set to None (use all data)

        Output:
            - data/metadata.csv with columns:
                patient_id, resolution, scan_path, cdr, age, sex,
                educ, mmse, diagnosis, label

        Statistics:
            Prints class distribution, age statistics, sex distribution
        """

    patient_pattern = os.path.join(BASE_FOLDER, "OAS1_*_MR*")
    patient_folders = glob(patient_pattern)
    metadata = []

    skipped_age = 0
    skipped_sex = 0
    skipped_cdr = 0

    for patient_folder in patient_folders:
        patient_id = os.path.basename(patient_folder)
        txt_file = os.path.join(patient_folder, f"{patient_id}.txt")

        if not os.path.exists(txt_file):
            continue

        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        #metadata
        #noinspection SpellCheckingInspection
        cdr = age = sex = educ = mmse = None

        for line in content.split('\n'):
            line = line.strip()
            if 'CDR:' in line:
                try:
                    cdr = float(line.split('CDR:')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'AGE:' in line:
                try:
                    age = int(line.split('AGE:')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif 'M/F:' in line:
                try:
                    sex = line.split('M/F:')[1].strip()
                except (ValueError, IndexError):
                    pass
            elif 'EDUC:' in line:
                try:
                    educ = int(line.split('EDUC:')[1].strip())
                except (ValueError, IndexError):
                    pass
            #noinspection SpellCheckingInspection
            elif 'MMSE:' in line:
                try:
                    # noinspection SpellCheckingInspection
                    mmse = int(line.split('MMSE:')[1].strip())
                except (ValueError, IndexError):
                    pass

        #demographic filters
        if cdr is None:
            skipped_cdr += 1
            continue

        if AGE_MIN is not None and (age is None or age < AGE_MIN): # type: ignore
            skipped_age += 1
            continue

        if AGE_MAX is not None and (age is None or age > AGE_MAX): # type: ignore
            skipped_age += 1
            continue

        if SEX_FILTER is not None and sex != SEX_FILTER:
            skipped_sex += 1
            continue

        if EDUC_MIN is not None and (educ is None or educ < EDUC_MIN): # type: ignore
            continue

        #labeling
        if cdr == 0:
            diagnosis = 'CN'  # Cognitively Normal
            label = 0
        elif cdr == 0.5:
            diagnosis = 'VeryMild'  # Very Mild Impairment (CDR 0.5)
            label = 1
        elif cdr >= 1.0:
            diagnosis = 'Dementia'  # Mild+ Dementia (CDR ≥1)
            label = 2
        else:
            continue

        #Find processed scans
        for res in [96, 128]:
            processed_folder = os.path.join(OUTPUT_BASE, f"processed_{res}")
            scan_pattern = os.path.join(processed_folder, f"{patient_id}_*.nii.gz")
            processed_scans = glob(scan_pattern)

            for scan_path in processed_scans:
                metadata.append({
                    'patient_id': patient_id,
                    'resolution': res,
                    'scan_filename': os.path.basename(scan_path),
                    'scan_path': scan_path,
                    'cdr': cdr,
                    'age': age,
                    'sex': sex,
                    'educ': educ,
                    'mmse': mmse,
                    'diagnosis': diagnosis,
                    'label': label
                })

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(OUTPUT_BASE, 'metadata.csv'), index=False)

    print(f"\n Saved {len(df)} scan entries")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"\nFilters applied:")
    print(f"Age range: {AGE_MIN or 'None'} - {AGE_MAX or 'None'}")
    print(f"Sex: {SEX_FILTER or 'Both'}")
    print(f"Skipped: {skipped_age} (age), {skipped_sex} (sex), {skipped_cdr} (no CDR)")

    print(f"\nClass distribution (patients):")
    print(df.groupby('diagnosis')['patient_id'].nunique())
    print(f"\nAge statistics:")
    print(df.groupby('diagnosis')['age'].describe())
    print(f"\nSex distribution:")
    print(pd.crosstab(df['diagnosis'], df['sex']))

#noinspection DuplicatedCode
def create_binary_split(df_binary, splits_base, res):
    """
        Create train/val/test splits for binary classification (CN vs VeryMild).

        Split strategy:
            - 60% train, 20% validation, 20% test
            - Stratified by class (maintains class balance)
            - Random seed 42 (reproducible)

        Args:
            df_binary: DataFrame with CN and VeryMild patients only
            splits_base: Base directory for splits (data/splits/)
            res: Resolution (96 or 128)

        Outputs:
            - data/splits/res_{res}_binary/train.csv
            - data/splits/res_{res}_binary/val.csv
            - data/splits/res_{res}_binary/test.csv
        """

    try:
        train_b, temp_b = train_test_split(df_binary, test_size=0.4, random_state=42,
                                          stratify=df_binary['binary_label'])
        val_b, test_b = train_test_split(temp_b, test_size=0.5, random_state=42,
                                        stratify=temp_b['binary_label'])

        split_dir = os.path.join(splits_base, f'res_{res}_binary')
        os.makedirs(split_dir, exist_ok=True)
        train_b.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
        val_b.to_csv(os.path.join(split_dir, 'val.csv'), index=False)
        test_b.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

        print(f"\nBinary (CN vs VeryMild):")
        print(f"Train: {len(train_b)}, Val: {len(val_b)}, Test: {len(test_b)}")
    except ValueError as e:
        print(f"Binary split failed: {e}")

#noinspection DuplicatedCode
def create_3class_split(df_res, splits_base, res):
    """
        Create train/val/test splits for 3-class classification.

        Same strategy as binary split but includes all 3 classes:
            - CN (CDR=0)
            - VeryMild (CDR=0.5)
            - Dementia (CDR≥1.0)

        Args:
            df_res: DataFrame with all patients
            splits_base: Base directory for splits
            res: Resolution (96 or 128)

        Outputs:
            - data/splits/res_{res}_3class/train.csv
            - data/splits/res_{res}_3class/val.csv
            - data/splits/res_{res}_3class/test.csv
        """

    try:
        train_3, temp_3 = train_test_split(df_res, test_size=0.4, random_state=42,
                                          stratify=df_res['label'])
        val_3, test_3 = train_test_split(temp_3, test_size=0.5, random_state=42,
                                        stratify=temp_3['label'])

        split_dir = os.path.join(splits_base, f'res_{res}_3class')
        os.makedirs(split_dir, exist_ok=True)
        train_3.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
        val_3.to_csv(os.path.join(split_dir, 'val.csv'), index=False)
        test_3.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

        print(f"\n  3-Class (CN vs VeryMild vs Dementia):")
        print(f"    Train: {len(train_3)}, Val: {len(val_3)}, Test: {len(test_3)}")
    except ValueError as e:
        print(f"  3-class split failed: {e}")


def create_splits():
    """
        Create train/val/test splits for all tasks and resolutions.

        For each resolution (96, 128):
            1. Load metadata.csv
            2. Keep only one scan per patient (first alphabetically)
            3. Create binary splits (CN vs VeryMild only)
            4. Create 3-class splits (CN vs VeryMild vs Dementia)

        Key Decision:
            One scan per patient prevents data leakage (same patient
            appearing in both train and test sets)

        Outputs:
            - data/splits/res_96_binary/{train,val,test}.csv
            - data/splits/res_96_3class/{train,val,test}.csv
            - data/splits/res_128_binary/{train,val,test}.csv
            - data/splits/res_128_3class/{train,val,test}.csv

        Statistics:
            Prints class distribution and sample counts for each split
        """

    df = pd.read_csv(os.path.join(OUTPUT_BASE, 'metadata.csv'))
    splits_base = os.path.join(OUTPUT_BASE, 'splits')

    for res in [96, 128]:
        df_res = df[df['resolution'] == res]

        #Keep 1 scan per patient
        df_res = df_res.sort_values('scan_filename').groupby('patient_id').first().reset_index()

        print(f"\nResolution {res}³:")
        print(f"  Total patients: {len(df_res)}")
        print(f"  Class balance: {df_res['diagnosis'].value_counts().to_dict()}")

        #Binary split: CN vs VeryMild (for early detection)
        df_binary = df_res[df_res['label'].isin([0, 1])].copy()
        df_binary['binary_label'] = df_binary['label']  # 0=CN, 1=VeryMild
        create_binary_split(df_binary, splits_base, res)

        #3-class split: CN vs VeryMild vs Dementia
        create_3class_split(df_res, splits_base, res)


if __name__ == '__main__':
    print("="*60)
    print("Right Preprocessing - Using OASIS Processed Data")
    print("="*60)
    print(f"Demographics filters:")
    print(f"  Age: {AGE_MIN or 'None'} - {AGE_MAX or 'None'}")
    print(f"  Sex: {SEX_FILTER or 'Both'}")
    print("="*60)

    process_all_resolutions()
    extract_labels_with_demographics()
    create_splits()

    print("\n" + "="*60)
    print("Ready for training")
    print("="*60)