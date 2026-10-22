"""
IMU 数据导入与预处理模块 (preprocess_v2)
=========================================

与 `preprocess/`（v1）的唯一实质差异：**整条流水线统一输出 100 Hz**。
v1 把信号插值到 `_align_imu` 估计出的设备原生率（~99.29 Hz，随设备浮动）；
v2 改为直接插值到固定的 100 Hz 栅格，使不同设备、不同批次的数据落在同一条
时间栅格上，可以逐行并列比较。

处理流程（相对 v1 少一次插值）：
1. 批量导入原始 CSV，解析片上时间与系统时间
2. 晶振频偏拟合：把片上采样计数器映射回主机墙钟（真实秒）
3. **唯一一次插值**：不规则的原始样本 → 100 Hz 等间隔栅格
   （同时完成"补丢帧"与"重采样"，v1 需两步）
4. 在 100 Hz 栅格上生成插值掩码
5. 20 Hz 低通滤波（4 阶 Butterworth，Nyquist 用 100 Hz）

输出 8 列：`time, ax, ay, az, wx, wy, wz, interp_mask`

命令行用法
----------
    python imu_processor.py <数据目录> <输出CSV> [--t-diff 6.56] [--no-lpf]

典型用法
--------
>>> from imu_processor import data_import, data_preprocess
>>> raw = data_import("path/to/csv_dir")
>>> proc, info = data_preprocess(raw, t_diff=6.56, return_drift=True)
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import butter, filtfilt

__version__ = "3.0.0"
__all__ = ["data_import", "data_preprocess", "FS_OUT", "INTERP_METHOD"]


# =============================================================================
# 全局常量：统一采样率与插值方式
# =============================================================================

FS_OUT = 100.0
"""float: 全流程统一输出采样率 (Hz)。整条流水线不存在其他频率硬编码。"""

INTERP_METHOD = "pchip"
"""str: 重采样插值方式，'pchip'（保形三次，默认）或 'linear'（线性）。

- 'pchip'：分段三次 Hermite，无过冲、保形，与 v1 行为一致，数值误差更小。
- 'linear'：分段线性，更直观、计算更快，但插值误差略大。

两者在 20 Hz 低通之后差异很小（实测相对 RMS 误差差别 < 0.5%），
默认沿用 v1 的 'pchip' 以保证输出与旧版可比。
"""

LPF_ORDER = 4
"""int: 低通滤波器阶数（Butterworth）。"""

_N_BAD_CHIP_GAP = 0.5
"""float: 片上时间戳异常判定阈值（秒）。正常间隔约 10 ms，BLE 损坏时间戳
会给出数百至数十万秒的跳变，必须在拟合前剔除。"""


# =============================================================================
# 内部辅助函数
# =============================================================================

def _time_transform(arr):
    """
    将时间字符串数组转换为秒。

    支持格式: HH:MM:SS.mmm 或 HH.MM.SS.mmm

    Parameters
    ----------
    arr : array-like of str

    Returns
    -------
    ndarray
        转换后的秒数
    """
    arr = np.array([t.replace(":", ".").split(".") for t in arr], dtype=float)
    return arr[:, 0] * 3600 + arr[:, 1] * 60 + arr[:, 2] + arr[:, 3] / 1000


def _make_grid(t0, t1, fs):
    """
    构造 [t0, t1] 区间内的等间隔目标时间轴。

    与 v1 的 ``np.arange(t0, t1 + dt/2, dt)`` 不同，这里严格不越过 t1，
    避免最后一个点落在原始数据范围之外而变成外推。

    Parameters
    ----------
    t0, t1 : float
        区间端点（秒），t1 >= t0
    fs : float
        目标采样率 (Hz)

    Returns
    -------
    ndarray
        长度 ``floor((t1-t0)*fs)+1`` 的等间隔时间轴
    """
    return t0 + np.arange(int(np.floor(max(t1 - t0, 0.0) * fs)) + 1) / fs


def _build_interpolator(t, y, method):
    """
    构造从不规则样本 (t, y) 到任意时间点的插值器。

    Parameters
    ----------
    t : ndarray, shape (n,)
        单调递增的原始时间轴
    y : ndarray, shape (n, C)
        原始信号，按 axis=0 插值
    method : str
        'pchip' 或 'linear'

    Returns
    -------
    callable
        接受新时间轴、返回插值结果的插值器
    """
    if method == "pchip":
        return PchipInterpolator(t, y, axis=0)
    if method == "linear":
        return interp1d(t, y, kind="linear", axis=0,
                        bounds_error=False, fill_value="extrapolate")
    raise ValueError(f"未知的插值方式: {method!r}（可选 'pchip' / 'linear'）")


def _align_imu(data, fs_out=FS_OUT, interp_method=INTERP_METHOD,
               gap_threshold=0.02):
    """
    片上时间 → 系统墙钟对齐 + 重采样到统一栅格 + 丢帧间隙检测。

    这是全流程中**唯一一次插值**：直接把不规则的原始样本
    （含丢帧造成的空洞）插值到 `fs_out` 等间隔栅格上，
    一次性完成 v1 中的"补帧"与"重采样"两步。

    Parameters
    ----------
    data : ndarray, shape (n, >=8)
        列0=片上时间, 列1=系统时间, 列2:8=信号(6通道)
    fs_out : float, default FS_OUT
        目标采样率 (Hz)
    interp_method : str, default INTERP_METHOD
        插值方式：'pchip' 或 'linear'
    gap_threshold : float, default 0.02
        判定丢帧间隙的相邻时间差阈值（秒）

    Returns
    -------
    t_new : ndarray, shape (m,)
        `fs_out` 等间隔目标时间轴（真实秒）
    out : ndarray, shape (m, 6)
        在 t_new 上插值后的 6 轴信号
    fs_native : float
        设备原生采样率估计值（~99.3 Hz），仅用于诊断与漂移报告
    drift : float
        标称 100 Hz 与实际原生率的偏差 (%)
    gaps : ndarray, shape (n_gaps, 2)
        丢帧间隙区间 [(gap_start, gap_end), ...]，无丢帧时为 (0, 2)
    """
    t_sys = np.asarray(data[:, 1], dtype=float)
    t_chip = np.asarray(data[:, 0], dtype=float)
    sig = data[:, 2:8].astype(np.float64)

    # 1. 清洗片上时间异常（BLE 损坏时间戳会让 polyfit 严重偏斜，必须剔除）
    dt = np.diff(t_chip)
    ok = np.concatenate(([True], (dt > 0) & (dt < _N_BAD_CHIP_GAP)))
    t_sys, t_chip, sig = t_sys[ok], t_chip[ok], sig[ok]

    # 2. 晶振频偏拟合：片上采样计数器 → 主机墙钟（真实秒）
    k, b = np.polyfit(t_sys, t_chip, 1)
    fs_native = 100.0 * k
    drift = (1.0 - k) * 100.0

    # 3. 反解平滑真实时间轴
    t_est = (t_chip - b) / k

    # 4. 唯一一次插值：不规则样本 → fs_out 等间隔栅格
    t_new = _make_grid(t_est[0], t_est[-1], fs_out)
    out = _build_interpolator(t_est, sig, interp_method)(t_new)

    # 5. 丢帧间隙（在原始样本时间轴上判定，区间供掩码使用）
    gap_i = np.nonzero(np.diff(t_est) >= gap_threshold)[0]
    gaps = (np.column_stack([t_est[gap_i], t_est[gap_i + 1]])
            if len(gap_i) else np.empty((0, 2)))

    return t_new, out, fs_native, drift, gaps


def _build_mask(timestamps, gaps):
    """
    根据丢帧间隙在指定时间轴上生成插值掩码。

    落在任一丢帧间隙 [gap_start, gap_end] 内的采样点标记为 1，其余为 0。

    Parameters
    ----------
    timestamps : ndarray, shape (n,)
        目标时间轴（秒）
    gaps : ndarray, shape (n_gaps, 2)
        丢帧间隙区间 [(gap_start, gap_end), ...]

    Returns
    -------
    ndarray, shape (n,)
        插值掩码，1=该点由丢帧间隙插值得到，0=正常采样点
    """
    mask = np.zeros(len(timestamps), dtype=float)
    for gap_start, gap_end in gaps:
        mask[(timestamps >= gap_start) & (timestamps <= gap_end)] = 1.0
    return mask


def _lowpass_6axis(data, fs, apply_lpf=True, lpf_fc=20.0):
    """
    对 6 轴 IMU 信号做低通滤波（Butterworth，零相位）。

    不做 EKF 姿态解算 / 重力分离。归一化截止频率由传入的 `fs` 决定；
    在 v2 中该值恒为 FS_OUT（100 Hz），20 Hz 截止对应 2.5 倍 Nyquist 余量。

    Parameters
    ----------
    data : ndarray, shape (n, 6)
        6 轴信号 [ax, ay, az, wx, wy, wz]
    fs : float
        信号采样率 (Hz)
    apply_lpf : bool
        是否应用低通滤波
    lpf_fc : float
        截止频率 (Hz)

    Returns
    -------
    ndarray, shape (n, 6)
    """
    if not apply_lpf or lpf_fc <= 0 or fs <= 2.0 * lpf_fc:
        return np.asarray(data, dtype=float)
    b, a = butter(LPF_ORDER, lpf_fc / (fs / 2.0), btype="low")
    return filtfilt(b, a, np.asarray(data, dtype=float), axis=0)


# =============================================================================
# 公共 API
# =============================================================================

def data_import(filedir, verbose=True):
    """
    从指定目录批量导入 IMU CSV 数据。

    读取目录下所有 .csv 文件（**按文件名排序**，保证跨文件时间连续），
    仅取每文件前 9 列（跳过表头行），解析片上时间与系统时间。

    Parameters
    ----------
    filedir : str
        包含 .csv 文件的目录路径。文件应为带表头的原始数据，
        前 9 列为：第0列系统时间、第1列设备名称、第2列片上时间戳
        （含日期前缀）、第3~8列为 6 轴 IMU 信号
        [加速度X, 加速度Y, 加速度Z, 角速度X, 角速度Y, 角速度Z]。
        第9列起的占位列（角度/磁场/四元数等，常含空格 ' '）被跳过。
    verbose : bool, default True
        是否打印导入信息

    Returns
    -------
    np.ndarray, shape (n_samples, 8)
        合并后的数据数组，各列为：
        - 第0列: 片上时间 (秒)
        - 第1列: 系统时间 (秒)
        - 第2~7列: 6 轴 IMU 信号 [ax, ay, az, wx, wy, wz]

    Examples
    --------
    >>> raw = data_import("F:/IMU_Data/Data1_83")
    >>> print(raw.shape)
    (461628, 8)
    """
    if not os.path.isdir(filedir):
        raise ValueError(f"目录不存在: {filedir}")

    files = sorted(f for f in os.listdir(filedir) if f.endswith(".csv"))
    if not files:
        raise ValueError(f"目录中未找到 .csv 文件: {filedir}")

    # 仅读取前 9 列；后续角度/磁场/四元数列含空格占位符，不属于 6 轴信号
    df = pd.concat(
        [pd.read_csv(os.path.join(filedir, f), header=None, skiprows=1,
                     usecols=list(range(9))) for f in files],
        ignore_index=True)

    if verbose:
        print(f"Loaded {len(files)} files, {len(df)} samples from {filedir}")

    return np.column_stack((_time_transform([x.split(" ", 2)[2] for x in df[2]]),
                            _time_transform(df[0]),
                            df.iloc[:, 3:].values))


def data_preprocess(data, t_diff, apply_lpf=True, lpf_fc=20.0,
                    gap_threshold=0.02, fs_out=FS_OUT,
                    interp_method=INTERP_METHOD, return_drift=False):
    """
    对原始 IMU 数据进行完整预处理（统一输出 100 Hz，8 列）。

    处理流程：
    1. 片上时间 → 系统墙钟对齐，剔除损坏时间戳，拟合晶振频偏
    2. 一次性插值到 `fs_out` 等间隔栅格（默认 100 Hz），同时补掉帧
    3. 时间偏移 t_diff（秒级全局偏移，丢弃 t < 0 的样本）
    4. 在该栅格上生成插值掩码
    5. `lpf_fc` 低通滤波（默认 20 Hz，4 阶 Butterworth，零相位）

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, >=8)
        原始数据，第0列片上时间、第1列系统时间、第2~7列 6 轴信号。
        通常直接使用 `data_import()` 的返回值。
    t_diff : float
        时间偏移量（秒），用于对齐视频等外部数据源。
    apply_lpf : bool, default True
        是否应用低通滤波。
    lpf_fc : float, default 20.0
        低通滤波截止频率 (Hz)。
    gap_threshold : float, default 0.02
        判定丢帧间隙的相邻时间差阈值（秒）。
    fs_out : float, default FS_OUT (100.0)
        统一输出采样率 (Hz)。修改此值即可整体切换到其他栅格。
    interp_method : str, default INTERP_METHOD ('pchip')
        重采样插值方式，'pchip' 或 'linear'。
    return_drift : bool, default False
        若为 True，额外返回诊断信息字典。

    Returns
    -------
    proc : np.ndarray, shape (m, 8)
        预处理后的数据，各列为：
        - 第0列: 对齐后的时间 (秒，等间隔 1/fs_out，t_diff 已扣除)
        - 第1~3列: 低通后加速度 [ax, ay, az] (g)
        - 第4~6列: 低通后角速度 [wx, wy, wz] (deg/s)
        - 第7列: 插值掩码 (1=丢帧间隙内的插值点, 0=正常采样点)
    info : dict, optional
        当 return_drift=True 时返回，含 'fs'（输出采样率）、
        'fs_native'（设备原生率估计）、'drift'（频偏 %）、
        'dropped_samples'（t_diff 丢弃点数）、
        'interpolated_samples'（掩码标记点数）。

    Examples
    --------
    >>> raw = data_import("path/to/data")
    >>> proc, info = data_preprocess(raw, t_diff=6.56, return_drift=True)
    >>> proc.shape, info['fs']
    ((466570, 8), 100.0)
    """
    # 1-2. 时间对齐 + 重采样到统一栅格
    t_new, out, fs_native, drift, gaps = _align_imu(
        data, fs_out=fs_out, interp_method=interp_method,
        gap_threshold=gap_threshold)

    # 3. 时间偏移（秒级全局偏移，丢弃负时间样本）
    ta = t_new - t_new[0] - t_diff
    keep = ta >= 0

    # 4-5. 插值掩码 + 低通滤波（均在统一栅格上）
    mask = _build_mask(t_new[keep], gaps)
    sig_f = _lowpass_6axis(out[keep], fs=fs_out, apply_lpf=apply_lpf,
                           lpf_fc=lpf_fc)

    result = np.column_stack((ta[keep], sig_f, mask))

    if return_drift:
        info = {"fs": fs_out,
                "fs_native": fs_native,
                "drift": drift,
                "dropped_samples": int((~keep).sum()),
                "interpolated_samples": int(mask.sum())}
        return result, info
    return result


# =============================================================================
# 命令行入口
# =============================================================================

def _main(argv=None):
    """CLI: 导入 → 预处理 → 保存 8 列 CSV。"""
    from imu_utils import save_processed_csv

    p = argparse.ArgumentParser(
        description="IMU 预处理 (preprocess_v2)：统一输出 100 Hz、8 列 CSV")
    p.add_argument("data_dir", help="原始 IMU CSV 所在目录")
    p.add_argument("output_csv", help="输出 CSV 路径")
    p.add_argument("--t-diff", type=float, default=0.0,
                   help="时间偏移（秒），用于对齐视频；默认 0")
    p.add_argument("--fs-out", type=float, default=FS_OUT,
                   help=f"统一输出采样率 (Hz)，默认 {FS_OUT:g}")
    p.add_argument("--interp", choices=("pchip", "linear"),
                   default=INTERP_METHOD,
                   help=f"重采样插值方式，默认 {INTERP_METHOD}")
    p.add_argument("--lpf-fc", type=float, default=20.0,
                   help="低通截止频率 (Hz)，默认 20")
    p.add_argument("--no-lpf", action="store_true", help="关闭低通滤波")
    p.add_argument("--gap-threshold", type=float, default=0.02,
                   help="丢帧间隙判定阈值（秒），默认 0.02")
    args = p.parse_args(argv)

    raw = data_import(args.data_dir, verbose=True)
    proc, info = data_preprocess(
        raw, t_diff=args.t_diff, apply_lpf=not args.no_lpf,
        lpf_fc=args.lpf_fc, gap_threshold=args.gap_threshold,
        fs_out=args.fs_out, interp_method=args.interp, return_drift=True)

    print(f"output   : {proc.shape[0]} samples x {proc.shape[1]} columns")
    print(f"fs_out   : {info['fs']:.6f} Hz   (device native "
          f"{info['fs_native']:.6f} Hz, drift {info['drift']:+.3f}%)")
    print(f"dropped  : {info['dropped_samples']}   "
          f"interpolated: {info['interpolated_samples']} "
          f"({100.0 * info['interpolated_samples'] / proc.shape[0]:.3f}%)")
    save_processed_csv(proc, args.output_csv)


if __name__ == "__main__":
    _main()
