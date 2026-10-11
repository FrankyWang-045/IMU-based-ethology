from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from data import Data
from utils import ROOT, load_config, output_dir


DROP_FIRST_SECONDS = 90.0
DROP_LAST_SECONDS = 60.0


def lowpass_filter(data, cutoff, fs, order=4):
    """对多变量时序数据做零相位 Butterworth 低通滤波。"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def extract_device_id(rec_name):
    """从 recording 名中提取设备 ID，例如 A1_C1_A1-6d -> 6d。"""
    if "-" in rec_name:
        return rec_name.rsplit("-", 1)[-1]
    return "unknown"


class DataPreprocess(Data):
    """IMU 预处理：对 raw 6 轴做低通滤波、降采样，并可选 per-device z-score。"""

    def __init__(
        self,
        file_path,
        preprocess_cfg,
        cutoff=25.0,
        order=4,
        flip_xz=None,
        device_stats=None,
        quiet=False,
    ):
        super().__init__(file_path)
        self.cutoff = cutoff
        self.order = order
        self.features = None
        self.cfg_preproc = preprocess_cfg
        self.flip_xz = set(flip_xz or [])
        self.device_stats = device_stats
        self.quiet = quiet

    def preprocess(self):
        raw_6axis = np.hstack([self.acc, self.gyro])

        # ---------- 对指定 recording 翻转 acc/gyro 的 X、Z 轴 ----------
        if self.name in self.flip_xz:
            if not self.quiet:
                print(f"  [flip X/Z] {self.name}")
            raw_6axis[:, [0, 2]] *= -1   # ax, az
            raw_6axis[:, [3, 5]] *= -1   # wx, wz
        # ----------------------------------------------------------------

        # ---------- 删除前 90 秒和后 60 秒，保留绝对时间戳 ----------
        t0 = self.time[0]
        t_end = self.time[-1]

        keep = (
            (self.time >= t0 + DROP_FIRST_SECONDS)
            & (self.time <= t_end - DROP_LAST_SECONDS)
        )

        if not keep.any():
            print(
                f"警告: {self.name} 有效长度不足 "
                f"{DROP_FIRST_SECONDS + DROP_LAST_SECONDS} 秒，无可用数据"
            )
            self.features = np.empty((0, raw_6axis.shape[1]), dtype=np.float32)
            self.time = self.time[keep]
            self.interp_mask = self.interp_mask[keep]
            return self

        raw_6axis = raw_6axis[keep]
        self.time = self.time[keep]
        self.interp_mask = self.interp_mask[keep]
        # -------------------------------------------------------------

        dt = np.median(np.diff(self.time))
        fs_orig = 1.0 / dt if dt > 0 else 100.0
        fs_target = self.cfg_preproc.get("target_fs", 25.0)

        # 抗混叠低通滤波
        cutoff = min(self.cutoff, fs_target / 2.0 - 1e-6)
        filtered = lowpass_filter(raw_6axis, cutoff, fs_orig, self.order)

        # 降采样
        factor = int(round(fs_orig / fs_target))
        if factor > 1:
            self.features = filtered[::factor]
            self.time = self.time[::factor]
            self.interp_mask = self.interp_mask[::factor]
        else:
            self.features = filtered

        # ---------- per-device z-score 归一化 ----------
        if self.device_stats is not None:
            dev = extract_device_id(self.name)
            if dev in self.device_stats:
                stat = self.device_stats[dev]
                self.features = (self.features - stat["mean"]) / stat["std"]
                if not self.quiet:
                    print(f"  [z-score {dev}] {self.name}")
            else:
                print(f"  警告: {self.name} 的设备 {dev} 无统计量，跳过 z-score")
        # ------------------------------------------------

        return self

    def save(self, out_dir):
        if self.features is None:
            raise RuntimeError("请先调用 preprocess() 再保存")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_dir / f"{self.name}.npz",
            features=self.features.astype(np.float32),
            time=self.time,
            interp_mask=self.interp_mask,
        )


def _compute_device_stats(cfg):
    """第一轮预处理：不降 z-score，累积每个设备的均值和方差。"""
    raw_dir = ROOT / cfg["project"]["raw_dir"]
    fc = cfg["preprocess"]
    flip_xz = fc.get("flip_xz_recordings", [])

    sums = {}
    sumsqs = {}
    counts = {}

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir} 中没有 csv 文件")

    print("=== 第一轮：计算 per-device 统计量 ===")
    for csv_path in csv_files:
        proc = DataPreprocess(
            csv_path,
            preprocess_cfg=fc,
            cutoff=fc.get("lowpass_cutoff", 25.0),
            order=fc.get("lowpass_order", 4),
            flip_xz=flip_xz,
            quiet=True,
        )
        proc.preprocess()
        if proc.features is None or len(proc.features) == 0:
            continue

        dev = extract_device_id(proc.name)
        feat = proc.features

        sums[dev] = sums.get(dev, 0.0) + feat.sum(axis=0)
        sumsqs[dev] = sumsqs.get(dev, 0.0) + (feat ** 2).sum(axis=0)
        counts[dev] = counts.get(dev, 0) + feat.shape[0]

    eps = fc.get("zscore_eps", 1e-8)
    stats = {}
    for dev in sums:
        mean = sums[dev] / counts[dev]
        var = sumsqs[dev] / counts[dev] - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0)) + eps
        stats[dev] = {"mean": mean.astype(np.float32),
                      "std": std.astype(np.float32),
                      "count": counts[dev]}
    return stats

def _save_device_stats(stats, out_dir):
    """把 per-device 统计量保存到 metadata 子目录，避免被 train.py 误加载。"""
    out_dir = Path(out_dir) / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / "device_zscore_stats.npz",
        **{f"{dev}_mean": s["mean"] for dev, s in stats.items()},
        **{f"{dev}_std":  s["std"]  for dev, s in stats.items()},
        **{f"{dev}_count": np.array(s["count"]) for dev, s in stats.items()},
        devices=np.array(list(stats.keys())),
    )

def preprocess_all(cfg):
    """批量预处理：raw_dir 下所有 csv -> output_dir 下的 npz。"""
    raw_dir = ROOT / cfg["project"]["raw_dir"]
    out = output_dir(cfg)
    fc = cfg["preprocess"]

    # 需要翻转 X/Z 轴的 recording 列表
    flip_xz = fc.get("flip_xz_recordings", [])

    per_device_zscore = fc.get("per_device_zscore", False)
    device_stats = _compute_device_stats(cfg) if per_device_zscore else None

    if per_device_zscore:
        print("已启用 per-device z-score，各设备统计：")
        for dev, s in sorted(device_stats.items()):
            print(f"  {dev}: count={s['count']}, "
                  f"mean={np.round(s['mean'], 4)}, "
                  f"std={np.round(s['std'], 4)}")
        _save_device_stats(device_stats, out)

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir} 中没有 csv 文件")

    print("=== 第二轮：归一化并保存 ===")
    for csv_path in csv_files:
        out_path = out / f"{csv_path.stem}.npz"

        # 启用 z-score 时强制重新生成（预处理配置已变化）
        if out_path.exists() and not per_device_zscore:
            print(f"跳过（已存在）: {csv_path.stem}")
            continue

        print(f"处理中: {csv_path.name}")
        proc = DataPreprocess(
            csv_path,
            preprocess_cfg=fc,
            cutoff=fc.get("lowpass_cutoff", 25.0),
            order=fc.get("lowpass_order", 4),
            flip_xz=flip_xz,
            device_stats=device_stats,
        )
        proc.preprocess()

        if proc.features is None or len(proc.features) == 0:
            print(f"  跳过（无有效数据）: {csv_path.stem}")
            continue

        proc.save(out)

    print("全部预处理完成")


if __name__ == "__main__":
    preprocess_all(load_config())