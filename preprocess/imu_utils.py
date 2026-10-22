"""
IMU 预处理辅助工具
====================

提供 notebook 流水线中使用的通用工具函数：
1. save_processed_csv — 将 data_preprocess() 输出保存为 CSV
2. t_diff_path        — t_diff 文件的标准路径 (与 aligner GUI 写入位置一致)
3. load_t_diff        — 读取 t_diff 文件里的数值
4. get_raw            — 按 IMU 目录缓存读取原始数据 (目录变化即自动重载)
5. clear_raw_cache    — 清空原始数据缓存

依赖: numpy (get_raw 内部惰性导入 imu_processor)
"""

import os

import numpy as np

__all__ = ["save_processed_csv", "t_diff_path", "load_t_diff",
           "get_raw", "clear_raw_cache"]

# t_diff 文件名 (与 aligner.IMUVideoAligner._save 写入的名字必须一致)
T_DIFF_FILENAME = "t_diff.txt"

# 按「规范化后的目录」缓存原始数据；键变化即代表换了 IMU 目录 → 自动重载。
# 这是修掉「一对小鼠文件完全相同」的关键：旧版用 `if "raw" not in globals()`
# 判断，只要 raw 还在就复用，换了 IMU_DIR 也不会重新读取 → 把上一只鼠的数据
# 写进了下一只鼠的输出文件。
_RAW_CACHE = {}


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


def t_diff_path(video_path, t_diff_file=None):
    """返回 t_diff 文件路径。

    默认取「视频同目录下的 t_diff.txt」——与 aligner GUI 的
    "Save t_diff" 写入位置一致。可用 t_diff_file 显式覆盖。
    """
    if t_diff_file:
        return os.path.abspath(t_diff_file)
    return os.path.join(os.path.dirname(os.path.abspath(video_path)),
                        T_DIFF_FILENAME)


def load_t_diff(video_path, t_diff_file=None):
    """读取 t_diff 文件并返回 float (秒)。

    DO_REGISTRATION=False 时 Cell 2 用它直接取配准值，跳过 GUI。
    """
    path = t_diff_path(video_path, t_diff_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到 t_diff 文件: {path}\n"
            f"请先运行配准 GUI 生成, 或在 Cell 0 设置 "
            f"T_DIFF_MANUAL / T_DIFF_FILE。")
    with open(path, "r") as f:
        txt = f.read().strip()
    if not txt:
        raise ValueError(f"t_diff 文件为空: {path}")
    # 只取第一个数值，容忍多余空白/换行
    return float(txt.split()[0])


def get_raw(imu_dir, force=False, verbose=True):
    """按 IMU 目录缓存读取原始数据。

    - 同一目录重复调用 → 复用缓存 (不重复读盘)
    - **目录变化** → 自动重新加载 (避免复用上一只鼠的旧数据)
    - force=True → 强制重新加载 (改完参数想确保刷新时用)

    Parameters
    ----------
    imu_dir : str
        IMU 原始 CSV 目录。
    force : bool
        True 时忽略缓存强制重新读取。
    verbose : bool
        是否打印来源/缓存信息。
    """
    key = os.path.normpath(os.path.abspath(imu_dir))
    if force or key not in _RAW_CACHE:
        from imu_processor import data_import   # 惰性导入, 避免循环依赖
        if verbose:
            print(f"[get_raw] {'强制重新' if force else '首次'}读取 {imu_dir}")
        _RAW_CACHE[key] = data_import(imu_dir, verbose=verbose)
    elif verbose:
        print(f"[get_raw] 复用缓存 ({imu_dir})")
    return _RAW_CACHE[key]


def clear_raw_cache():
    """清空原始数据缓存 (一般不必要; 需要彻底重读时调用)。"""
    _RAW_CACHE.clear()
