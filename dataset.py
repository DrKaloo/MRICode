import torch
from torch.utils.data import Dataset
import nibabel as nib
import pandas as pd
import os

class BrainMRIDataset(Dataset):

    def __init__(self, csv_file, transform=None, augment=False):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scan_path = row['scan_path'].replace('/', os.sep)
        label = int(row['label'])

        img = nib.load(scan_path)
        data = img.get_fdata() # type: ignore
        data = torch.FloatTensor(data).unsqueeze(0)

        #Enhanced augmentation to prevent overfitting
        if self.augment:
            data = self.augment_3d(data)

        if self.transform:
            data = self.transform(data)

        return data, label

    def augment_3d(self, x):
        #Random flip (left-right)
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])

        #Random rotation (small angles)
        if torch.rand(1) > 0.5:
            angle = (torch.rand(1) - 0.5) * 20  # ±10 degrees
            x = self.rotate_3d(x, angle)

        #Random noise
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise

        #Random intensity shift
        if torch.rand(1) > 0.5:
            shift = (torch.rand(1) - 0.5) * 0.2
            x = x + shift

        return x

    #def rotate_3d(self, x, angle):
        #Simple rotation around z-axis
        # Simplified just returning x for now
        # A good rotation requires more complex transforms
        return x

class RandomFlip3D:
    def __call__(self, x):
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
        return x