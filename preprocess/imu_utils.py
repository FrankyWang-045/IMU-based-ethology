"""
IMU 预处理辅助工具
====================

提供 notebook 流水线中使用的通用工具函数：
1. save_processed_csv — 将 data_preprocess() 输出保存为 CSV

依赖: numpy
"""

import os
import numpy as np

__all__ = ["save_processed_csv"]


def save_processed_csv(proc, filepath):
    """
    将 data_preprocess() 输出的 8 通道数据保存为 CSV 文件。

    Parameters
    ----------
    proc : np.ndarray, shape (m, 8)
        预处理后的数据，各列为：
        time, ax, ay, az, wx, wy, wz, interp_mask
    filepath : str
        输出 CSV 文件路径，如 "output/processed.csv"

    Returns
    -------
    str
        实际保存的文件路径

    Examples
    --------
    >>> proc = data_preprocess(raw, t_diff=6.56)
    >>> save_processed_csv(proc, "F:/data/processed_data2.csv")
    """
    proc = np.asarray(proc, dtype=float)
    if proc.ndim != 2 or proc.shape[1] != 8:
        raise ValueError(f"期望 shape (m, 8) 的数组，实际收到 {proc.shape}")

    out_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(out_dir, exist_ok=True)

    header = "time,ax,ay,az,wx,wy,wz,interp_mask"
    # %.17g 保证双精度浮点完整往返精度，不截断任何信息
    np.savetxt(filepath, proc, delimiter=",", header=header,
               comments="", fmt="%.17g")
    print(f"Saved {proc.shape[0]} samples x {proc.shape[1]} channels -> {filepath}")
    return filepath
