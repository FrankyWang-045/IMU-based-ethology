"""VAE 训练数据集：输入为滤波后的 6 轴 raw IMU。"""

import numpy as np
import torch
from torch.utils.data import Dataset


def load_raw_x(npz_path):
    """VAE 输入 (N, 6): [ax, ay, az, wx, wy, wz]。"""
    data = np.load(npz_path)
    return data["features"].astype(np.float32)


def compute_stats(npz_paths):
    """用全训练文件计算全局标准化统计量，返回 (mean, std)，形状 (6,)。"""
    x_all = np.concatenate([load_raw_x(p) for p in npz_paths], axis=0)
    return x_all.mean(axis=0), x_all.std(axis=0)


class PoseDataset(Dataset):
    """单个文件的随机切块数据集。"""

    def __init__(self, npz_path, chunk=1000, mean=None, std=None):
        x_raw = load_raw_x(npz_path)

        if mean is None:
            mean = x_raw.mean(axis=0)
        if std is None:
            std = x_raw.std(axis=0)
        self.x_mean = mean
        self.x_std = std

        self.x = ((x_raw - mean) / std).astype(np.float32)
        self.chunk = chunk
        self.n_frames = len(self.x)

    def __len__(self):
        return self.n_frames // self.chunk

    def __getitem__(self, idx):
        start = np.random.randint(0, self.n_frames - self.chunk)
        block = self.x[start : start + self.chunk]
        return torch.from_numpy(block)