"""
IMU 数据导入与预处理模块
============================

基于片上时间与系统时间对齐的 IMU 数据流水线，包含：
1. 多 CSV 文件批量导入与时间解析
2. 晶振频偏校正与帧丢失检测
3. 掉帧插值并生成插值掩码（丢帧间隙内的点标记为 1）
4. 时间对齐后保持 ~99Hz 均匀采样（非整数秒时间戳，不做上采样）
5. 20Hz 低通滤波（Butterworth），输出 6 轴 + 时间 + 插值掩码（8 列）

典型用法
--------
>>> from imu_processor import data_import, data_preprocess
>>> raw = data_import("path/to/csv_dir")
>>> proc = data_preprocess(raw, t_diff=6.56)   # 输出 ~99Hz, 8 列
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt

__version__ = "2.0.0"
__all__ = ["data_import", "data_preprocess"]


# =============================================================================
# 内部辅助函数
# =============================================================================

def _time_transform(arr):
    """
    将时间字符串数组转换为秒。
    支持格式: HH:MM:SS.mmm 或 HH.MM.SS.mmm
    """
    arr = np.array([t.replace(":", ".").split(".") for t in arr], dtype=float)
    secs = arr[:, 0] * 3600 + arr[:, 1] * 60 + arr[:, 2] + arr[:, 3] / 1000
    return secs


def _frameloss_test(t, threshold=0.02, frame_interval=0.01, verbose=True):
    """
    检测时间序列中的帧丢失。

    Parameters
    ----------
    t : array-like
        时间戳序列（秒）
    threshold : float
        判定帧丢失的时间差阈值（秒）
    frame_interval : float
        期望的帧间隔（秒）
    verbose : bool
        是否打印结果

    Returns
    -------
    fl_count : int
        帧丢失事件次数
    lostframes : int
        累计丢失帧数
    """
    t = np.asarray(t, dtype=float)
    fl_count = 0
    lostframes = 0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt >= threshold:
            fl_count += 1
            lostframes += int(dt / frame_interval) - 1
    if verbose:
        print(f"frame loss count: {fl_count}")
        print(f"total lost frames: {lostframes}")
    return fl_count, lostframes


def _align_imu(data, gap_threshold=0.02):
    """
    片上时间与系统时间对齐 + 掉帧插值 + 插值掩码生成

    Parameters
    ----------
    data : ndarray
        列0=片上时间, 列1=系统时间, 列2:8=信号(6通道)
    gap_threshold : float, default=0.02
        判定丢帧间隙的时间差阈值（秒），与 _frameloss_test 的阈值一致

    Returns
    -------
    t_new : ndarray
        等间隔目标时间轴（~99Hz 均匀，非整数秒）
    out : ndarray
        在 t_new 上插值后的信号 (n, 6)
    sig : ndarray
        清洗后的原始信号 (n, 6)
    fs : float
        估计采样频率 (~99Hz)
    drift : float
        晶振频偏 (%)
    t_est : ndarray
        反解后的平滑真实时间轴
    gaps : ndarray, shape (n_gaps, 2)
        丢帧间隙区间 [(gap_start, gap_end), ...]，
        间隙由相邻原始采样时间差 >= gap_threshold 判定。
        无丢帧时为空数组 (0, 2)。
    """
    # 强制数值化，避免字符串输入
    t_sys = np.array(data[:, 1], dtype=float)
    t_chip = np.array(data[:, 0], dtype=float)

    sig = data[:, 2:8].astype(np.float64)

    # 1. 清洗片上时间异常
    dt = np.diff(t_chip)
    ok = np.concatenate(([True], (dt > 0) & (dt < 0.5)))
    t_sys, t_chip, sig = t_sys[ok], t_chip[ok], sig[ok]

    # 2. 晶振频偏拟合
    k, b = np.polyfit(t_sys, t_chip, 1)
    fs = 100.0 * k
    drift = (1.0 - k) * 100

    # 3. 反解平滑真实时间轴
    t_est = (t_chip - b) / k

    # 4. 等间隔目标轴
    dt = 1.0 / fs
    t_new = np.arange(t_est[0], t_est[-1] + dt / 2, dt)

    out = np.zeros((len(t_new), sig.shape[1]))
    for i in range(sig.shape[1]):
        pchip = PchipInterpolator(t_est, sig[:, i])
        out[:, i] = pchip(t_new)

    # 5. 检测丢帧间隙（用于生成插值掩码）
    dt_est = np.diff(t_est)
    gap_idx = np.nonzero(dt_est >= gap_threshold)[0]
    if len(gap_idx) > 0:
        gaps = np.column_stack([t_est[gap_idx], t_est[gap_idx + 1]])
    else:
        gaps = np.empty((0, 2))

    return t_new, out, sig, fs, drift, t_est, gaps


def _build_mask(timestamps, gaps):
    """
    根据丢帧间隙在指定时间轴上生成插值掩码。

    落在任一丢帧间隙 [gap_start, gap_end] 内的采样点标记为 1，
    其余为 0。可用于任意采样率的时间轴（插值轴、重采样轴等）。

    Parameters
    ----------
    timestamps : ndarray, shape (n,)
        目标时间轴（秒）
    gaps : ndarray, shape (n_gaps, 2)
        丢帧间隙区间 [(gap_start, gap_end), ...]

    Returns
    -------
    mask : ndarray, shape (n,)
        插值掩码，1=该点由丢帧间隙插值得到，0=正常采样点
    """
    mask = np.zeros(len(timestamps), dtype=float)
    for gap_start, gap_end in gaps:
        mask[(timestamps >= gap_start) & (timestamps <= gap_end)] = 1.0
    return mask


def _lowpass_6axis(data, fs, apply_lpf=True, lpf_fc=20.0):
    """
    对 6 轴 IMU 信号做低通滤波（4 阶 Butterworth）。

    不做 EKF 姿态解算 / 重力分离（与早期粗分类版本不同）。
    滤波器的 Nyquist 与阶跃响应均使用 _align_imu 估计出的真实 fs，
    而非上采样后的频率，因此可直接在 ~99Hz 信号上以 20Hz 截止工作。

    Parameters
    ----------
    data : ndarray, shape (n, 6)
        6 轴信号 [ax, ay, az, wx, wy, wz]
    fs : float
        实际采样频率 (Hz)，取自 _align_imu 估计值（~99）
    apply_lpf : bool
        是否应用低通滤波
    lpf_fc : float
        低通滤波截止频率 (Hz)

    Returns
    -------
    ndarray, shape (n, 6)
        滤波后的 6 轴信号
    """
    data = np.asarray(data, dtype=float)
    if not apply_lpf or lpf_fc <= 0:
        return data
    # 仅在截止频率低于 Nyquist 一半时设计滤波器，
    # 避免 fs 过低时归一化频率 >= 1 导致数值异常。
    if fs <= 2.0 * lpf_fc:
        return data
    nyq = fs / 2.0
    b, a = butter(4, lpf_fc / nyq, btype='low')
    return filtfilt(b, a, data, axis=0)


# =============================================================================
# 公共 API
# =============================================================================

def data_import(filedir, verbose=True):
    """
    从指定目录批量导入 IMU CSV 数据。

    读取目录下所有 .csv 文件，仅取每文件前 9 列（跳过表头行），
    解析片上时间与系统时间，并检测帧丢失情况。

    Parameters
    ----------
    filedir : str
        包含 .csv 文件的目录路径。文件应为带表头的原始数据，
        前 9 列为：第0列系统时间、第1列设备名称、第2列片上时间戳
        （含日期前缀）、第3~8列为 6 轴 IMU 信号
        [加速度X, 加速度Y, 加速度Z, 角速度X, 角速度Y, 角速度Z]。
        第9列起的占位列（角度/磁场/四元数等，常含空格 ' '）被跳过。
    verbose : bool, default=True
        是否打印帧丢失统计信息。

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
    (100000, 8)
    """
    if not os.path.isdir(filedir):
        raise ValueError(f"目录不存在: {filedir}")

    files = [f for f in os.listdir(filedir) if f.endswith('.csv')]
    if not files:
        raise ValueError(f"目录中未找到 .csv 文件: {filedir}")

    df_list = []
    for f in files:
        # 仅读取原始数据前 9 列: 系统时间(0)、设备名称(1)、片上时间(2)、
        # 加速度 X/Y/Z(3~5)、角速度 X/Y/Z(6~8)。
        # 后续角度/磁场/四元数等列含空格占位符 (' ')，不属于 6 轴信号，跳过。
        # skiprows=1 跳过表头行; usecols 仅取前 9 列。
        df = pd.read_csv(os.path.join(filedir, f), header=None,
                         skiprows=1, usecols=list(range(9)))
        df_list.append(df)

    data = pd.concat(df_list, ignore_index=True)

    # 解析时间
    time_onchip = _time_transform([x.split(" ", 2)[2] for x in data[2]])
    time_sys = _time_transform(data[0])

    if verbose:
        print(f"Loaded {len(files)} files, {len(data)} samples from {filedir}")
    _frameloss_test(time_onchip, verbose=verbose)

    return np.concatenate((time_onchip.reshape(-1, 1),
                           time_sys.reshape(-1, 1),
                           data.iloc[:, 3:].values), axis=1)


def data_preprocess(data, t_diff, apply_lpf=True, lpf_fc=20.0,
                    gap_threshold=0.02, return_drift=False):
    """
    对原始 IMU 数据进行完整预处理（无上采样，输出 ~99Hz）。

    处理流程：
    1. 片上时间与系统时间对齐，校正晶振频偏（_align_imu，逻辑保持不变）
    2. 时间偏移 t_diff（秒级全局偏移，丢弃 t < 0 的样本）
    3. 在 ~99Hz 均匀时间轴上生成插值掩码（丢帧间隙内的点标记为 1）
    4. 20Hz 低通滤波（作用于 6 轴信号，不做 EKF / 重力分离）

    输出采样率 ≈ 由 _align_imu 估计的片上采样率（~99Hz，非整数秒时间戳）。

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, >=8)
        原始数据，第0列为片上时间，第1列为系统时间，
        第2~7列为6轴 IMU 信号 [ax, ay, az, wx, wy, wz]。
        通常直接使用 `data_import()` 的返回值。
    t_diff : float
        时间偏移量（秒），用于对齐视频等外部数据源。
    apply_lpf : bool, default=True
        是否应用低通滤波。
    lpf_fc : float, default=20.0
        低通滤波截止频率 (Hz)。
    gap_threshold : float, default=0.02
        判定丢帧间隙的时间差阈值（秒），用于生成插值掩码。
    return_drift : bool, default=False
        若为 True，额外返回晶振漂移信息字典。

    Returns
    -------
    proc : np.ndarray, shape (m, 8)
        预处理后的数据，各列为：
        - 第0列: 对齐后的时间 (秒，非整数秒，~99Hz 均匀)
        - 第1~3列: 滤波后加速度 [ax, ay, az] (g)
        - 第4~6列: 滤波后角速度 [wx, wy, wz] (deg/s)
        - 第7列: 插值掩码 (1=丢帧间隙内插值点, 0=正常采样点)
    info : dict, optional
        当 return_drift=True 时返回，包含 'fs'、'drift'、
        'dropped_samples'、'interpolated_samples' 字段。

    Examples
    --------
    >>> raw = data_import("path/to/data")
    >>> proc = data_preprocess(raw, t_diff=6.56)
    >>> proc.shape
    (99900, 8)
    """
    # 1. 时间对齐与插值（保持原逻辑不变，输出 ~99Hz 均匀信号 out 与时间轴 t_new）
    t_new, out, sig, fs, drift, t_est, gaps = _align_imu(data, gap_threshold=gap_threshold)

    # 2. 时间偏移（秒级全局偏移，丢弃负时间样本）
    t_new_0 = t_new - t_new[0]
    ta = t_new_0 - t_diff
    keep = ta >= 0
    ta_p = ta[keep]
    imu_p = out[keep]                      # out 已是 (n, 6) 的 ~99Hz 均匀信号

    # 3. 在 ~99Hz 时间轴上生成插值掩码（丢帧间隙内的点标记为 1）
    mask_full = _build_mask(t_new, gaps)
    mask_p = mask_full[keep]

    # 4. 低通滤波（去掉 EKF / 重力分离，仅做 20Hz 低通）
    imu_f = _lowpass_6axis(imu_p, fs=fs, apply_lpf=apply_lpf, lpf_fc=lpf_fc)

    result = np.concatenate([ta_p.reshape(-1, 1),
                             imu_f,
                             mask_p.reshape(-1, 1)], axis=1)

    if return_drift:
        info = {'fs': fs, 'drift': drift,
                'dropped_samples': int((~keep).sum()),
                'interpolated_samples': int(mask_p.sum())}
        return result, info
    return result
