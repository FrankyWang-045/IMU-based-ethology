import numpy as np
import pandas as pd
from data import Data
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PoseDataset(Dataset):
    def __init__(self, npz_path, chunk=1000):
        data = np.load(npz_path)
        feats = data["features"]          # (N, 9)
        trig = feats[:, 9:13]             # 通道正弦/余弦值
        wz = feats[:, 5:6]                  # gyro z
        print(trig.shape)
        self.wz_mean = wz.mean()
        self.wz_std = wz.std()
        wz_zscore = (wz - self.wz_mean) / self.wz_std
        x_raw = np.hstack([trig, wz])                  # (N, 5)，wz 不再单独处理
        self.x_mean = x_raw.mean(axis=0)               # (5,) 逐轴均值
        self.x_std = x_raw.std(axis=0)                 # (5,) 逐轴标准差
        self.x = ((x_raw - self.x_mean) / self.x_std).astype(np.float32)
        self.chunk = chunk
        self.n_frames = len(self.x)


    def __len__(self):
        return self.n_frames // self.chunk

    def __getitem__(self, idx):
        start = np.random.randint(0, self.n_frames - self.chunk)
        block = self.x[start : start + self.chunk]     # (chunk, 5)
        return torch.from_numpy(block)

