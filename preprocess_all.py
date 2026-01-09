import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import zoom, binary_erosion, binary_dilation, binary_fill_holes
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_FOLDER = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data\raw"
OUTPUT_BASE = r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\data"

#demographics
# noinspection PyTypeChecker
AGE_MIN = None
AGE_MAX = None
SEX_FILTER = None  #'M', 'F', or None
EDUC_MIN = None


def preprocess_scan_proper(input_path, output_path, target_shape):

    img = nib.load(input_path)
    data = img.get_fdata() # type: ignore
    original_affine = img.affine # type: ignore
    #(line above) keeping original spatial information/geometry - preserving spatial information

    if data.ndim == 4:
        data = data.squeeze()

    # Better brain mask using percentile-based threshold
    threshold = np.percentile(data[data > 0], 10)
    brain_mask = data > threshold

    # Morphological cleanup
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = binary_erosion(brain_mask, iterations=1)
    brain_mask = binary_dilation(brain_mask, iterations=2)

    #Intensity normalization
    brain_voxels = data[brain_mask]
    if len(brain_voxels) < 100:
        return None

    # Use percentile normalization (more robust than mean/std)
    p01, p99 = np.percentile(brain_voxels, [1, 99])
    data_clipped = np.clip(data, p01, p99)

# noinspection DuplicatedCode
    # Z-score normalization
    mean_val = brain_voxels.mean()
    std_val = brain_voxels.std()
    data_normalized = (data_clipped - mean_val) / (std_val + 1e-8)
    data_normalized[~brain_mask] = 0

    # Crop image to brain
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

    # High quality resampling
    zoom_factors = [target_shape[i] / data_cropped.shape[i] for i in range(3)]
    data_resized = zoom(data_cropped, zoom_factors, order=3)  # Cubic interpolation

    # Create proper affine for resized image
    # Adjust affine to account for cropping and resampling
    crop_translation = np.array([x_min, y_min, z_min])
    new_affine = original_affine.copy()
    new_affine[:3, 3] += original_affine[:3, :3] @ crop_translation

    # Adjust for resampling
    zoom_matrix = np.diag([1/zoom_factors[0], 1/zoom_factors[1], 1/zoom_factors[2], 1])
    new_affine = new_affine @ zoom_matrix

    # Save with proper affine
    output_img = nib.Nifti1Image(data_resized, affine=new_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(output_img, output_path)

    return target_shape


def find_processed_scans():
    patient_folders = glob(os.path.join(BASE_FOLDER, "OAS1_*_MR*"))
    scan_files = []

    for patient_folder in patient_folders:
        #better quality first once processed
        # noinspection SpellCheckingInspection
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

        print(f"✓ Completed {res}³: {success}/{len(scan_files)} successful")


def extract_labels_with_demographics():
    """
    Extract labels AND demographics, apply filters
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
        # noinspection SpellCheckingInspection
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
            # noinspection SpellCheckingInspection
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
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"\nFilters applied:")
    print(f"  Age range: {AGE_MIN or 'None'} - {AGE_MAX or 'None'}")
    print(f"  Sex: {SEX_FILTER or 'Both'}")
    print(f"  Skipped: {skipped_age} (age), {skipped_sex} (sex), {skipped_cdr} (no CDR)")

    print(f"\nClass distribution (patients):")
    print(df.groupby('diagnosis')['patient_id'].nunique())
    print(f"\nAge statistics:")
    print(df.groupby('diagnosis')['age'].describe())
    print(f"\nSex distribution:")
    print(pd.crosstab(df['diagnosis'], df['sex']))

# noinspection DuplicatedCode
def create_binary_split(df_binary, splits_base, res):
    """Helper to create binary classification splits"""
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

        print(f"\n  Binary (CN vs VeryMild):")
        print(f"    Train: {len(train_b)}, Val: {len(val_b)}, Test: {len(test_b)}")
    except ValueError as e:
        print(f"  Binary split failed: {e}")

# noinspection DuplicatedCode
def create_3class_split(df_res, splits_base, res):
    """Helper to create 3-class splits"""
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
    df = pd.read_csv(os.path.join(OUTPUT_BASE, 'metadata.csv'))
    splits_base = os.path.join(OUTPUT_BASE, 'splits')

    for res in [96, 128]:
        df_res = df[df['resolution'] == res]

        # Keep 1 scan per patient
        df_res = df_res.sort_values('scan_filename').groupby('patient_id').first().reset_index()

        print(f"\nResolution {res}³:")
        print(f"  Total patients: {len(df_res)}")
        print(f"  Class balance: {df_res['diagnosis'].value_counts().to_dict()}")

        # Binary split: CN vs VeryMild (for early detection)
        df_binary = df_res[df_res['label'].isin([0, 1])].copy()
        df_binary['binary_label'] = df_binary['label']  # 0=CN, 1=VeryMild
        create_binary_split(df_binary, splits_base, res)

        # 3-class split: CN vs VeryMild vs Dementia
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