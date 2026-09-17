from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray
    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

def fit_standardizer(X: np.ndarray) -> Standardizer:
    return Standardizer(mean=X.mean(axis=0), std=X.std(axis=0))

class TabularCascadeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], standardizer: Optional[Standardizer] = None):
        X = df[feature_cols].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.float32)
        if standardizer is not None:
            X = standardizer.transform(X).astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).view(-1, 1)
    def __len__(self) -> int:
        return int(self.X.shape[0])
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
