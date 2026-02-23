import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    def __init__(self, csv_path, transform=None, cache_size=0):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.cache = {}
        self.cache_size = int(cache_size)

        required = {"scan_path", "label"}
        if not required.issubset(self.df.columns):
            raise RuntimeError(f"csv missing required columns: {required}")

    def __len__(self):
        return len(self.df)

    def _load_nifti(self, path):
        img = nib.load(path)
        x = img.get_fdata(dtype=np.float32)
        if x.ndim != 3:
            raise RuntimeError(f"Invalid volume shape: {x.shape}")
        x = np.expand_dims(x, axis=0)  # [1, D, H, W]
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["scan_path"]
        y = int(row["label"])

        if path in self.cache:
            x = self.cache[path]
        else:
            x = self._load_nifti(path)
            if self.cache_size > 0 and len(self.cache) < self.cache_size:
                self.cache[path] = x

        x = torch.from_numpy(x)

        if self.transform is not None:
            x = self.transform(x)

        return {"x": x, "y": y}
