import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import os

"""
PyTorch Dataset for 3D Brain MRI Classification

Loads preprocessed NIfTI brain scans from CSV file and converts to PyTorch tensors.
Supports optional data augmentation via transforms.

Classes:
    BrainMRIDataset: Main dataset class for loading MRI scans
    RandomFlip3D: Simple horizontal flip augmentation
"""

class BrainMRIDataset(Dataset):
    """
        Custom Dataset for loading 3D brain MRI scans from NIfTI files.

        Reads scan paths and labels from CSV file, loads NIfTI images using nibabel,
        and applies optional transforms for data augmentation.

        Args:
            csv_file (str): Path to CSV file containing scan information
            transform (callable, optional): Transform to apply to loaded scans (e.g., RandomFlip3D)
            augment (bool): Enable advanced augmentation (currently unused, default: False)

        CSV Format Expected:
            Required columns:
                - scan_path: Full path to .nii.gz file
                - label: Integer class label (0=CN, 1=VeryMild, etc.)
            Optional columns:
                - age, sex, educ, mmse, diagnosis, patient_id

        Returns:
            tuple: (data, label) where:
                - data: torch.Tensor of shape [1, H, W, D] (single channel 3D volume)
                - label: int (class label)

        Example:
            dataset = BrainMRIDataset('data/splits/res_128_binary/train.csv',
            ...                           transform=RandomFlip3D())
            data, label = dataset[0]
            print(data.shape)  # torch.Size([1, 128, 128, 128])
        """

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scan_path = row['scan_path'].replace('/', os.sep)
        label = int(row['label'])

        img = nib.load(scan_path)
        data = img.get_fdata() # type: ignore
        data = torch.FloatTensor(data).unsqueeze(0)

        if self.transform:
            data = self.transform(data)

        return data, label

class RandomFlip3D:
    def __call__(self, x):
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
        return x


"""
    Randomly flip 3D MRI scan horizontally (left-right) with 50% probability.

    This augmentation is anatomically valid for brain MRI as the left and right
    hemispheres are approximately symmetric. Helps prevent overfitting by creating
    mirror images of training samples.

    Args:
        None

    Returns:
        torch.Tensor: Flipped or original tensor (unchanged shape)

    Example:
        transform = RandomFlip3D()
        train_dataset = BrainMRIDataset('train.csv', transform=transform)
    """