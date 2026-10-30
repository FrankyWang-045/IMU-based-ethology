import numpy as np
import torch
from torch.utils.data import Dataset


def load_x(npz_path):
    """加载 6D 特征 [T, 6]。"""
    data = np.load(npz_path)
    return data["features"].astype(np.float32)


def compute_stats(npz_paths):
    x_all = np.concatenate([load_x(p) for p in npz_paths], axis=0)
    return x_all.mean(axis=0), x_all.std(axis=0)


class PoseAEDataset(Dataset):
    def __init__(self, npz_path, chunk=2048, mean=None, std=None):
        self.x_raw = load_x(npz_path)

        if mean is None:
            mean = self.x_raw.mean(axis=0)
        if std is None:
            std = self.x_raw.std(axis=0)
        self.x_mean = mean
        self.x_std = std + 1e-8

        self.chunk = chunk
        self.n_frames = len(self.x_raw)

    def __len__(self):
        return max(1, self.n_frames // self.chunk)

    def __getitem__(self, idx):
        if self.n_frames <= self.chunk:
            x = self.x_raw
        else:
            start = np.random.randint(0, self.n_frames - self.chunk)
            x = self.x_raw[start : start + self.chunk]

        x = ((x - self.x_mean) / self.x_std).astype(np.float32)
        return torch.from_numpy(x)