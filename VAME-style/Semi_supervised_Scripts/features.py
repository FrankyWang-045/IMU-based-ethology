"""按 HSMM 片段从原始 6 轴数据中提取特征。

参考 markmap 中的 4 类特征：
1. 时域统计
2. 头部姿态与朝向
3. 时频特征
4. 方向性与跨轴耦合
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import signal, stats
from scipy.integrate import cumulative_trapezoid

from utils import load_config, output_dir, latest_experiment_dir


def resolve_exp_dir(args_exp):
    cfg = load_config()
    out = output_dir(cfg)
    if args_exp:
        p = Path(args_exp)
        return p if p.is_absolute() else out / p
    return latest_experiment_dir(cfg)


def find_segments(time, states):
    """把 state 序列拆成连续片段。"""
    segs = []
    t = 0
    n = len(states)
    while t < n:
        s = t
        st = int(states[t])
        while t < n and states[t] == st:
            t += 1
        segs.append((st, time[s], time[t - 1], s, t))
    return segs


def compute_posture_heading(acc, gyro, fs):
    """从加速度计/陀螺仪估计姿态和朝向变化。"""
    ax, ay, az = acc[:, 0], acc[:, 1], acc[:, 2]
    wx, wy, wz = gyro[:, 0], gyro[:, 1], gyro[:, 2]

    # 简单倾斜估计（假设低动态时加速度主导重力）
    g = np.sqrt(ax**2 + ay**2 + az**2) + 1e-10
    pitch = np.arcsin(np.clip(-ay / g, -1, 1))  # 绕 x 轴
    roll = np.arctan2(ax, az)                    # 绕 y 轴

    # 航向变化：对陀螺仪 z 轴积分
    heading = cumulative_trapezoid(wz, dx=1.0 / fs, initial=0.0)
    heading_diff = heading[-1] - heading[0]
    heading_var = np.sum(np.abs(np.diff(heading)))

    return {
        "pitch_mean": np.mean(pitch),
        "pitch_std": np.std(pitch),
        "roll_mean": np.mean(roll),
        "roll_std": np.std(roll),
        "heading_diff": heading_diff,
        "heading_var": heading_var,
    }


def compute_band_powers(x, fs, bands):
    """计算频段功率。"""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    psd = np.maximum(psd, 1e-12)
    total = np.sum(psd)
    feats = {"total_power": total, "peak_freq": freqs[np.argmax(psd)]}

    # 谱重心
    feats["spectral_centroid"] = np.sum(freqs * psd) / total if total > 0 else 0.0

    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        feats[f"power_{name}"] = np.sum(psd[mask])
        feats[f"ratio_{name}"] = feats[f"power_{name}"] / total if total > 0 else 0.0

    return feats


def compute_time_domain(x):
    """单轴时域统计特征。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {k: 0.0 for k in [
            "mean", "median", "std", "var", "iqr", "range", "max", "min",
            "p2p", "absmax", "rms", "mav", "skew", "kurt"
        ]}

    q25, q75 = np.percentile(x, [25, 75])
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "std": np.std(x),
        "var": np.var(x),
        "iqr": q75 - q25,
        "range": np.max(x) - np.min(x),
        "max": np.max(x),
        "min": np.min(x),
        "p2p": np.max(x) - np.min(x),
        "absmax": np.max(np.abs(x)),
        "rms": np.sqrt(np.mean(x**2)),
        "mav": np.mean(np.abs(x)),
        "skew": float(stats.skew(x)) if len(x) > 2 else 0.0,
        "kurt": float(stats.kurtosis(x)) if len(x) > 2 else 0.0,
    }


def extract_segment_features(raw, fs):
    """对一个 6 轴片段 (T, 6) 提取全部特征，返回特征向量 + 名称列表。"""
    if raw.shape[0] < 5:
        return None, None

    acc = raw[:, :3]
    gyro = raw[:, 3:6]
    axis_names = ["ax", "ay", "az", "wx", "wy", "wz"]

    feats = {}
    names = []

    # 1. 时域统计：每轴 14 个特征
    for i, name in enumerate(axis_names):
        td = compute_time_domain(raw[:, i])
        for k, v in td.items():
            feats[f"td_{name}_{k}"] = v
            names.append(f"td_{name}_{k}")

    # 2. 姿态与朝向
    ph = compute_posture_heading(acc, gyro, fs)
    for k, v in ph.items():
        feats[f"posture_{k}"] = v
        names.append(f"posture_{k}")

    # 3. 时频特征：每轴
    bands = {
        "trot": (5.0, 7.0),
        "sniff": (8.5, 11.0),
        "shake": (16.0, 20.0),
    }
    for i, name in enumerate(axis_names):
        if len(raw[:, i]) >= 64:
            tf = compute_band_powers(raw[:, i], fs, bands)
            for k, v in tf.items():
                feats[f"tf_{name}_{k}"] = v
                names.append(f"tf_{name}_{k}")
        else:
            for k in ["total_power", "peak_freq", "spectral_centroid",
                      "power_trot", "ratio_trot", "power_sniff", "ratio_sniff",
                      "power_shake", "ratio_shake"]:
                feats[f"tf_{name}_{k}"] = 0.0
                names.append(f"tf_{name}_{k}")

    # 4. 跨轴耦合
    corr_pairs = [
        ("ax", "ay", 0, 1), ("ax", "az", 0, 2), ("ay", "az", 1, 2),
        ("wx", "wy", 3, 4), ("wx", "wz", 3, 5), ("wy", "wz", 4, 5),
        ("ax", "wx", 0, 3), ("ay", "wy", 1, 4), ("az", "wz", 2, 5),
    ]
    for n1, n2, i1, i2 in corr_pairs:
        c = np.corrcoef(raw[:, i1], raw[:, i2])[0, 1] if len(raw) > 1 else 0.0
        feats[f"corr_{n1}_{n2}"] = float(c)
        names.append(f"corr_{n1}_{n2}")

    # 陀螺-加速度跨轴互相关（ grooming 签名：wx vs wz）
    if len(gyro) > 1:
        cc = np.correlate(gyro[:, 0] - gyro[:, 0].mean(),
                          gyro[:, 2] - gyro[:, 2].mean(),
                          mode="full")
        feats["crosscorr_wx_wz_max"] = float(np.max(np.abs(cc))) / (np.std(gyro[:, 0]) * np.std(gyro[:, 2]) * len(gyro) + 1e-10)
    else:
        feats["crosscorr_wx_wz_max"] = 0.0
    names.append("crosscorr_wx_wz_max")

    vec = np.array([feats[n] for n in names], dtype=float)
    return vec, names


def main():
    parser = argparse.ArgumentParser(description="HSMM 片段特征提取")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验目录，默认最新 exp")
    parser.add_argument("--min_duration", type=float, default=0.2,
                        help="最短片段时长（秒）")
    parser.add_argument("--suffix", type=str, default="default",
                        help="输出后缀")
    args = parser.parse_args()

    cfg = load_config()
    fs = cfg["preprocess"]["target_fs"]
    out = output_dir(cfg)
    exp_dir = resolve_exp_dir(args.exp)

    print(f"实验目录: {exp_dir}")
    print(f"原始数据目录: {out}")

    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))
    if not hsmm_files:
        raise FileNotFoundError(f"{exp_dir} 中没有 *_hsmm.npz")

    all_features = []
    all_meta = []
    feature_names = None

    for p in hsmm_files:
        rec = p.stem.replace("_hsmm", "")
        raw_path = out / f"{rec}.npz"

        if not raw_path.exists():
            print(f"  跳过（无原始数据）: {rec}")
            continue

        print(f"处理: {rec}")

        hsmm = np.load(p)
        raw = np.load(raw_path)
        time_h = hsmm["time"]
        states = hsmm["states"]
        time_r = raw["time"]
        features_r = raw["features"]

        # 对齐原始数据到 HSMM 时间轴
        indices = np.searchsorted(time_r, time_h)
        indices = np.clip(indices, 0, len(time_r) - 1)
        features_aligned = features_r[indices]

        segments = find_segments(time_h, states)
        for state, t0, t1, s, e in segments:
            dur = t1 - t0
            if dur < args.min_duration:
                continue

            seg_raw = features_aligned[s:e]
            vec, names = extract_segment_features(seg_raw, fs)
            if vec is None:
                continue

            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise ValueError("特征名称不一致")

            all_features.append(vec)
            all_meta.append({
                "recording": rec,
                "state": int(state),
                "start": float(t0),
                "end": float(t1),
                "duration": float(dur),
                "n_frames": int(e - s),
            })

    if not all_features:
        raise RuntimeError("没有提取到任何特征")

    X = np.array(all_features, dtype=float)
    print(f"提取片段数: {len(X)}")
    print(f"特征维度: {X.shape[1]}")

    out_path = exp_dir / f"segment_features_{args.suffix}.npz"
    np.savez_compressed(
        out_path,
        features=X,
        feature_names=np.array(feature_names),
        meta=np.array(all_meta, dtype=object),
    )
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()