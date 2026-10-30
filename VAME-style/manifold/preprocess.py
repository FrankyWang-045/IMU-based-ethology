from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from scipy.stats import wasserstein_distance

from utils import load_config, output_dir


def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def rotation_cost(features, ref_features, gravity_weight=2.0):
    cost = 0.0
    for i in range(features.shape[1]):
        f = features[:, i]
        r = ref_features[:, i]
        scale = np.std(np.concatenate([f, r]))
        w = gravity_weight if i >= 6 else 1.0
        if scale > 1e-9:
            cost += w * wasserstein_distance(f / scale, r / scale)
        else:
            cost += w * abs(f.mean() - r.mean())
    return cost


def fibonacci_sphere(n_points):
    idx = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * idx / n_points)
    theta = np.pi * (1 + np.sqrt(5)) * idx
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def extract_middle_segment(arr, n_samples):
    if len(arr) <= n_samples:
        return arr
    start = (len(arr) - n_samples) // 2
    return arr[start : start + n_samples]


class Timer:
    """简易上下文计时器，收集各步骤耗时。"""

    def __init__(self, records, name):
        self.records = records
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.t0
        self.records[self.name] = self.records.get(self.name, 0.0) + elapsed


class Data:
    def __init__(self, file_path):
        self.path = Path(file_path)
        self.name = self.path.stem
        df = pd.read_csv(self.path)
        self.time = df["time"].to_numpy()
        self.acc = df[["ax", "ay", "az"]].to_numpy()
        self.gyro = df[["wx", "wy", "wz"]].to_numpy()
        self.interp_mask = df["interp_mask"].to_numpy()


class DataPreprocess(Data):
    def __init__(self, file_path, cfg, quiet=False):
        super().__init__(file_path)
        self.cfg = cfg
        self.quiet = quiet
        self.features = None
        self.R0 = None

    def _prepare(self):
        raw = np.hstack([self.acc, self.gyro])

        if self.name in set(self.cfg.get("flip_xz_recordings", [])):
            if not self.quiet:
                print(f"  [flip X/Z] {self.name}")
            raw[:, [0, 2]] *= -1
            raw[:, [3, 5]] *= -1

        t0, t_end = self.time[0], self.time[-1]
        keep = (self.time >= t0 + self.cfg["drop_first_seconds"]) & (
            self.time <= t_end - self.cfg["drop_last_seconds"]
        )
        if not keep.any():
            return np.empty((0, 6), dtype=np.float32)

        raw = raw[keep]
        self.time = self.time[keep]
        self.interp_mask = self.interp_mask[keep]

        return raw

    def _madgwick_separation(self, acc, gyro, fs):
        try:
            from ahrs.filters import Madgwick
        except ImportError:
            raise ImportError("请安装 ahrs：pip install ahrs")

        if self.cfg.get("acc_unit", "g") == "m/s2":
            acc_g = acc / 9.80665
        else:
            acc_g = acc.copy()

        if self.cfg.get("gyro_unit", "deg/s") == "deg/s":
            gyro_rad = np.deg2rad(gyro)
        else:
            gyro_rad = gyro.copy()

        beta = self.cfg.get("madgwick_beta", 0.1)
        madgwick = Madgwick(gyr=gyro_rad, acc=acc_g, frequency=fs, beta=beta)
        quaternions = madgwick.Q

        gravity = np.zeros_like(acc)
        for t in range(len(acc)):
            q = quaternions[t]
            R_sensor_to_world = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            gravity[t] = R_sensor_to_world.T @ np.array([0.0, 0.0, 1.0])

        if self.cfg.get("acc_unit", "g") == "m/s2":
            gravity = gravity * 9.80665

        linear_acc = acc - gravity
        return linear_acc, gravity

    def _search_best_rotation(self, features_9, ref_features_9, fs):
        minutes = self.cfg.get("rotation_segment_minutes", 5)
        n_samples = int(minutes * 60 * fs)

        feat_mid = extract_middle_segment(features_9, n_samples)
        ref_mid = extract_middle_segment(ref_features_9, n_samples)

        linear_acc = features_9[:, :3]
        gyro = features_9[:, 3:6]
        gravity = features_9[:, 6:9]

        g = gravity.mean(axis=0)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-6:
            raise RuntimeError(f"{self.name}: 无法估计重力方向")
        g = g / g_norm

        ref_g = ref_features_9[:, 6:9].mean(axis=0)
        ref_g = ref_g / np.linalg.norm(ref_g)

        n_sphere = self.cfg.get("sphere_samples", 10)
        sphere_dirs = fibonacci_sphere(n_sphere)
        targets = np.vstack([ref_g[None, :], -ref_g[None, :], sphere_dirs])

        twist_step = self.cfg.get("twist_step", 30.0)
        twist_angles = np.deg2rad(np.arange(0, 360, twist_step))

        best_cost = np.inf
        best_features = features_9
        best_R = R.from_matrix(np.eye(3))

        for target in targets:
            R_base, _ = R.align_vectors(target[None, :], g[None, :])
            for twist in twist_angles:
                R_twist = R.from_rotvec(twist * target)
                R_total = R_twist * R_base

                la_rot = R_total.apply(linear_acc)
                g_rot = R_total.apply(gyro)
                gr_rot = R_total.apply(gravity)
                rot_features = np.hstack([la_rot, g_rot, gr_rot])

                rot_mid = extract_middle_segment(rot_features, n_samples)
                cost = rotation_cost(rot_mid, ref_mid)
                if cost < best_cost:
                    best_cost = cost
                    best_features = rot_features
                    best_R = R_total

        self.R0 = best_R
        if not self.quiet:
            euler = best_R.as_euler("zyx", degrees=True)
            print(f"  [sphere search] {self.name}: euler={np.round(euler, 1)}, cost={best_cost:.4f}")
        return best_features

    def preprocess(self, ref_features_9=None, records=None):
        records = records if records is not None else {}

        with Timer(records, "load_prepare"):
            raw = self._prepare()
        if len(raw) == 0:
            self.features = np.empty((0, 9), dtype=np.float32)
            return self

        acc = raw[:, :3]
        gyro = raw[:, 3:]

        dt = np.median(np.diff(self.time))
        fs_orig = 1.0 / dt if dt > 0 else 100.0

        with Timer(records, "madgwick"):
            if self.cfg.get("use_madgwick", True):
                linear_acc, gravity = self._madgwick_separation(acc, gyro, fs_orig)
            else:
                g_mean = acc.mean(axis=0)
                g_mean = g_mean / np.linalg.norm(g_mean)
                gravity = np.tile(g_mean, (len(acc), 1))
                linear_acc = acc - gravity
        features_9 = np.hstack([linear_acc, gyro, gravity])

        with Timer(records, "rotation_search"):
            if ref_features_9 is not None:
                features_9 = self._search_best_rotation(features_9, ref_features_9, fs_orig)
            else:
                self.R0 = R.from_matrix(np.eye(3))

        with Timer(records, "lowpass"):
            cutoff = min(self.cfg["lowpass_cutoff"], fs_orig / 2.0 - 1e-6)
            self.features = lowpass_filter(features_9, cutoff, fs_orig, self.cfg["lowpass_order"])

        return self

    def save(self, out_dir, records=None):
        records = records if records is not None else {}
        with Timer(records, "save"):
            if self.features is None:
                raise RuntimeError("请先调用 preprocess()")
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / f"{self.name}.npz",
                features=self.features.astype(np.float32),
                time=self.time,
                interp_mask=self.interp_mask,
                R0=self.R0.as_matrix().astype(np.float32),
            )
        return records


def print_timing_summary(name, records, total):
    print(f"\n--- {name} 耗时诊断 ---")
    for k, v in sorted(records.items(), key=lambda x: -x[1]):
        pct = 100 * v / total if total > 0 else 0
        print(f"  {k:20s}: {v:8.3f}s  ({pct:5.1f}%)")
    print(f"  {'total':20s}: {total:8.3f}s")


def preprocess_all(cfg):
    raw_dir = Path(cfg["project"]["raw_dir"])
    out = output_dir(cfg)
    cfg_p = cfg["preprocess"]
    reference_name = cfg_p.get("reference_name")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir} 中没有 csv 文件")

    total_records = {}
    t_start_all = time.perf_counter()

    ref_features_9 = None
    if reference_name:
        t0 = time.perf_counter()
        ref_csv = raw_dir / f"{reference_name}.csv"
        if not ref_csv.exists():
            raise ValueError(f"参考 recording {reference_name} 未找到")
        print(f"=== 参考 recording: {reference_name} ===")
        ref_proc = DataPreprocess(ref_csv, cfg_p)
        ref_proc.preprocess(ref_features_9=None, records=total_records)
        ref_features_9 = ref_proc.features
        ref_proc.save(out, records=total_records)
        print_timing_summary(reference_name, total_records, time.perf_counter() - t0)

    print("\n=== 批量预处理：Madgwick -> 球面旋转搜索（中段 5min）-> 低通 ===")
    for csv_path in csv_files:
        if csv_path.stem == reference_name:
            continue

        out_path = out / f"{csv_path.stem}.npz"
        if out_path.exists():
            print(f"跳过（已存在）: {csv_path.stem}")
            continue

        t0 = time.perf_counter()
        file_records = {}
        print(f"处理中: {csv_path.name}")
        proc = DataPreprocess(csv_path, cfg_p)
        proc.preprocess(ref_features_9=ref_features_9, records=file_records)
        if len(proc.features) == 0:
            print(f"  跳过（无有效数据）: {csv_path.stem}")
            continue
        proc.save(out, records=file_records)
        elapsed = time.perf_counter() - t0
        print_timing_summary(csv_path.stem, file_records, elapsed)

        # 累加到总耗时
        for k, v in file_records.items():
            total_records[k] = total_records.get(k, 0.0) + v

    total_elapsed = time.perf_counter() - t_start_all
    print("\n========== 总耗时诊断 ==========")
    print_timing_summary("ALL RECORDINGS", total_records, total_elapsed)


if __name__ == "__main__":
    preprocess_all(load_config())