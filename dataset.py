import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    """
    Loads preprocessed NIfTI volumes listed in a CSV.

    Required CSV columns:
      - scan_path
      - label

    Optional:
      - subject_id
      - patient_id
      - age
      - sex
      - split

    Output:
      - if return_dict=True: {"x": Tensor[C=1,D,H,W], "y": int, "meta": dict}
      - else: (x, y)

    cache_size:
      - 0 disables caching
      - >0 keeps an in-memory LRU cache of volumes to avoid reloading .nii.gz every epoch
    """

    def __init__(
        self,
        csv_file: str,
        transform=None,
        return_dict: bool = True,
        enforce_float32: bool = True,
        nan_to_num: bool = True,
        cache_size: int = 0,
        strict_shape: bool = True,
    ):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.return_dict = return_dict
        self.enforce_float32 = enforce_float32
        self.nan_to_num = nan_to_num
        self.strict_shape = strict_shape

        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

        for col in ["scan_path", "label"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        self.df["scan_path"] = self.df["scan_path"].astype(str).map(os.path.normpath)

        # Set to prefer subject_id as patient_id if present (stable subject identity)
        if "patient_id" not in self.df.columns:
            if "subject_id" in self.df.columns:
                self.df["patient_id"] = self.df["subject_id"].astype(str)
            else:
                self.df["patient_id"] = self.df["scan_path"].apply(
                    lambda p: os.path.basename(p).split(".nii")[0]
                )

        if "age" not in self.df.columns:
            self.df["age"] = np.nan
        if "sex" not in self.df.columns:
            self.df["sex"] = np.nan
        if "split" not in self.df.columns:
            self.df["split"] = np.nan

    def __len__(self) -> int:
        return len(self.df)

    def _postprocess_volume(self, vol: np.ndarray, scan_path: str) -> np.ndarray:
        if vol.ndim == 4:
            vol = np.squeeze(vol)

        if self.strict_shape and vol.ndim != 3:
            raise ValueError(f"Expected 3D volume after squeeze, got shape={vol.shape} from {scan_path}")

        if self.nan_to_num:
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

        if self.enforce_float32 and vol.dtype != np.float32:
            vol = vol.astype(np.float32, copy=False)

        return np.ascontiguousarray(vol)

    def _load_volume(self, scan_path: str) -> np.ndarray:
        # LRU cache hit
        if self.cache_size > 0 and scan_path in self._cache:
            vol = self._cache.pop(scan_path)
            self._cache[scan_path] = vol
            return vol

        img = nib.load(scan_path)
        vol = img.get_fdata(dtype=np.float32 if self.enforce_float32 else None)
        vol = self._postprocess_volume(vol, scan_path)

        # LRU cache insert
        if self.cache_size > 0:
            self._cache[scan_path] = vol
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return vol

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Dict[str, Any]]:
        row = self.df.iloc[idx]
        scan_path = str(row["scan_path"])
        label = int(row["label"])

        vol = self._load_volume(scan_path)  # numpy [X,Y,Z]

        # Convert to torch [C=1, D, H, W] to match torchvision video models [B,C,D,H,W]
        # nibabel typical order is (X,Y,Z); treat Z as depth (D)
        vol_t = torch.from_numpy(vol).permute(2, 0, 1).contiguous()  # [D,H,W]
        x = vol_t.unsqueeze(0)  # [1,D,H,W]

        if self.transform is not None:
            x = self.transform(x)

        if self.return_dict:
            meta = {
                "patient_id": str(row["patient_id"]),
                "scan_path": scan_path,
                "age": None if pd.isna(row["age"]) else float(row["age"]),
                "sex": None if pd.isna(row["sex"]) else str(row["sex"]),
                "split": None if pd.isna(row["split"]) else str(row["split"]),
            }
            return {"x": x, "y": label, "meta": meta}

        return x, label