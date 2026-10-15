"""训练数据集：返回 (x, x_low, x_high)，其中 x_low 为因果 1Hz 低通。"""

import numpy as np
import torch
from scipy.signal import butter, sosfilt
from scipy.spatial.transform import Rotation
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


def random_so3_augment(x, max_angle_deg=15.0, p=1.0):
    """
    对 (T, 6) 的 IMU 数据做随机 SO(3) 旋转。
    x[:, 0:3] = [ax, ay, az], x[:, 3:6] = [wx, wy, wz]。
    acc 与 gyro 使用同一旋转矩阵。
    """
    if p < 1.0 and np.random.rand() > p:
        return x

    max_angle = np.deg2rad(max_angle_deg)
    theta = np.random.uniform(0.0, max_angle)

    # 在球面上均匀采样旋转轴
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    R = Rotation.from_rotvec(theta * axis).as_matrix()

    x_aug = x.copy()
    x_aug[:, :3] = x[:, :3] @ R.T   # acc
    x_aug[:, 3:] = x[:, 3:] @ R.T   # gyro
    return x_aug


class PoseDataset(Dataset):
    def __init__(
        self,
        npz_path,
        chunk=2048,
        mean=None,
        std=None,
        lowpass_cutoff=1.0,
        fs=25.0,
        augment=False,
        aug_max_angle_deg=15.0,
        aug_p=1.0,
    ):
        self.x_raw = load_raw_x(npz_path)

        if mean is None:
            mean = self.x_raw.mean(axis=0)
        if std is None:
            std = self.x_raw.std(axis=0)
        self.x_mean = mean
        self.x_std = std + 1e-8

        self.chunk = chunk
        self.n_frames = len(self.x_raw)
        self.lowpass_cutoff = lowpass_cutoff
        self.fs = fs

        self.augment = augment
        self.aug_max_angle_deg = aug_max_angle_deg
        self.aug_p = aug_p

    def __len__(self):
        return self.n_frames // self.chunk

    def __getitem__(self, idx):
        start = np.random.randint(0, self.n_frames - self.chunk)
        x = self.x_raw[start : start + self.chunk].copy()

        # -------- SO(3) 数据增强：在全局 z-score 之前 --------
        if self.augment:
            x = random_so3_augment(x, self.aug_max_angle_deg, self.aug_p)
        # ----------------------------------------------------

        # 因果低通 + 高频残差
        x_low = causal_lowpass(x, cutoff=self.lowpass_cutoff, fs=self.fs)
        x_high = x - x_low

        # 全局 z-score
        x = ((x - self.x_mean) / self.x_std).astype(np.float32)
        x_low = ((x_low - self.x_mean) / self.x_std).astype(np.float32)
        x_high = (x_high / self.x_std).astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_low),
            torch.from_numpy(x_high),
        )