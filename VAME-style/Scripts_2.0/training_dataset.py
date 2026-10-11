"""训练数据集：返回 (x, x_low, x_high)，其中 x_low 为因果 1Hz 低通。"""

import numpy as np
import torch
from scipy.signal import butter, sosfilt
from torch.utils.data import Dataset


def load_raw_x(npz_path):
    """VAE 输入 (N, 6): [ax, ay, az, wx, wy, wz]。"""
    data = np.load(npz_path)
    return data["features"].astype(np.float32)


def compute_stats(npz_paths):
    x_all = np.concatenate([load_raw_x(p) for p in npz_paths], axis=0)
    return x_all.mean(axis=0), x_all.std(axis=0)


def causal_lowpass(x, cutoff=1.0, fs=25.0, order=4):
    """
    因果 Butterworth 低通滤波。
    x: (T, C)
    返回 x_low: (T, C)
    """
    if x.shape[0] <= order * 3:
        return x.copy()
    sos = butter(order, cutoff / (fs / 2.0), btype="low", output="sos")
    return sosfilt(sos, x, axis=0)


class PoseDataset(Dataset):
    def __init__(self, npz_path, chunk=2048, mean=None, std=None, lowpass_cutoff=1.0, fs=25.0):
        x_raw = load_raw_x(npz_path)

        # 因果低通 + 残差
        x_low = causal_lowpass(x_raw, cutoff=lowpass_cutoff, fs=fs)
        x_high = x_raw - x_low

        if mean is None:
            mean = x_raw.mean(axis=0)
        if std is None:
            std = x_raw.std(axis=0)
        self.x_mean = mean
        self.x_std = std + 1e-8

        self.x = ((x_raw - mean) / std).astype(np.float32)
        self.x_low = ((x_low - mean) / std).astype(np.float32)
        self.x_high = ((x_high) / std).astype(np.float32)

        self.chunk = chunk
        self.n_frames = len(self.x)

    def __len__(self):
        return self.n_frames // self.chunk

    def __getitem__(self, idx):
        start = np.random.randint(0, self.n_frames - self.chunk)
        x = self.x[start : start + self.chunk]
        x_low = self.x_low[start : start + self.chunk]
        x_high = self.x_high[start : start + self.chunk]
        return (
            torch.from_numpy(x),
            torch.from_numpy(x_low),
            torch.from_numpy(x_high),
        )