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
        wz_mean = wz.mean()
        wz_std = wz.std()
        wz_zscore = (wz - wz_mean) / wz_std
        self.x = np.hstack([trig, wz_zscore]).astype(np.float32)
        self.chunk = chunk
        self.n_frames = len(self.x)


    def __len__(self):
        return self.n_frames // self.chunk

    def __getitem__(self, idx):
        start = np.random.randint(0, self.n_frames - self.chunk)
        block = self.x[start : start + self.chunk]     # (chunk, 5)
        return torch.from_numpy(block)


ds = PoseDataset("output/A5_C5_C5-c8.npz", chunk=1000)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

for batch in loader:
    print(batch.shape)   # torch.Size([32, 1000, 5])
    break