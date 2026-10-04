"""对照实验：生成"原始特征版"的 downstream（绕过 VAE）。

从预处理 npz 提取 5 轴原始特征 -> 全体文件统一标准化 -> 裁预热帧对齐
-> 保存到 output_raw/ 目录，文件名与 latent 组一致。
"""

from pathlib import Path

import numpy as np

from training_dataset import load_raw_x, compute_stats
from utils import ROOT, load_config

SRC_DIR = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\output")
FILES = ["A4_B4_A4-1c", "A4_B4_B4-ab"]     # 与 latent 组一致的两只小鼠
WARMUP = 24                                 # 与 VAE 感受野对齐（25-1）


def main():
    cfg = load_config()
    out_raw = ROOT / "output_raw"
    out_raw.mkdir(exist_ok=True)

    src_paths = [SRC_DIR / f"{name}.npz" for name in FILES]

    # 全体文件统一统计量（与 latent 组的 compute_stats 同口径）
    mean, std = compute_stats(src_paths)

    for p in src_paths:
        data = np.load(p)
        x = (load_raw_x(p) - mean) / std

        downstream = x[WARMUP:].astype(np.float32)
        time = data["time"][WARMUP:]
        interp_mask = data["interp_mask"][WARMUP:]

        np.savez_compressed(out_raw / f"{p.stem}_latent.npz",
                            downstream=downstream,
                            time=time, interp_mask=interp_mask)
        print(f"{p.stem}: {downstream.shape}")

    print(f"对照组已生成到 {out_raw}")


if __name__ == "__main__":
    main()