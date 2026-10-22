from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R

from data import Data
from utils import ROOT, load_config, output_dir


DROP_FIRST_SECONDS = 90.0
DROP_LAST_SECONDS = 60.0


def lowpass_filter(data, cutoff, fs, order=4):
    """对多变量时序数据做零相移 Butterworth 低通滤波。"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


class DataPreprocess(Data):
    """IMU 预处理：
    1. 读取 RAW
    2. 可选 flip_xz
    3. 删除前后 90s/60s
    4. 静态安装角标定：找静止段，估计重力方向，整体旋转到世界坐标系
    5. 低通滤波
    6. 降采样
    7. 保存 features（物理单位，未归一化）
    """

    def __init__(
        self,
        file_path,
        preprocess_cfg,
        cutoff=25.0,
        order=4,
        flip_xz=None,
        quiet=False,
    ):
        super().__init__(file_path)
        self.cutoff = cutoff
        self.order = order
        self.features = None
        self.cfg_preproc = preprocess_cfg
        self.flip_xz = set(flip_xz or [])
        self.quiet = quiet

        self.use_gravity_align = preprocess_cfg.get("use_gravity_align", True)
        self.stationary_window_sec = preprocess_cfg.get("stationary_window_sec", 2.0)
        self.gravity_target = np.array(
            preprocess_cfg.get("gravity_target", [0.0, 0.0, 1.0])
        )

    def _find_stationary_window(self, acc, time):
        """用加速度方差最小的窗口估计静止段方向。"""
        if len(acc) == 0:
            return 0, 0

        fs = 1.0 / np.median(np.diff(time))
        window = int(self.stationary_window_sec * fs)
        window = max(window, 10)
        if window >= len(acc):
            return 0, len(acc)

        step = max(1, window // 4)
        min_var = np.inf
        best_start = 0
        for start in range(0, len(acc) - window + 1, step):
            var = np.var(acc[start : start + window], axis=0).sum()
            if var < min_var:
                min_var = var
                best_start = start
        return best_start, best_start + window

    def _static_gravity_alignment(self, acc, gyro, time):
        """静态安装角标定：找静止段，计算 device->world 旋转 R0。

        返回：
            acc_w: 世界系加速度
            gyro_w: 世界系角速度
            R0: scipy Rotation，device -> world
            g_dir: 估计的设备系重力方向
        """
        start, end = self._find_stationary_window(acc, time)
        g_dir = acc[start:end].mean(axis=0)
        g_norm = np.linalg.norm(g_dir)
        if g_norm < 1e-6:
            raise RuntimeError(f"{self.name}: 静止段加速度模长为 0，无法估计重力方向")
        g_dir = g_dir / g_norm

        # 把设备系下的 g_dir 对齐到世界重力方向
        R0, _ = R.align_vectors(self.gravity_target[None, :], g_dir[None, :])

        acc_w = R0.apply(acc)
        gyro_w = R0.apply(gyro)

        if not self.quiet:
            print(f"  [static align] {self.name}: g_dir={np.round(g_dir, 3)}")

        return acc_w, gyro_w, R0, g_dir

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

        acc = raw_6axis[:, :3]
        gyro = raw_6axis[:, 3:]

        # ---------- 静态重力对齐 ----------
        if self.use_gravity_align:
            acc, gyro, _, _ = self._static_gravity_alignment(acc, gyro, self.time)
        # ---------------------------------

        raw_6axis = np.hstack([acc, gyro])

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


def preprocess_all(cfg):
    """批量预处理：raw_dir 下所有 csv -> output_dir 下的 npz。"""
    raw_dir = ROOT / cfg["project"]["raw_dir"]
    out = output_dir(cfg)
    fc = cfg["preprocess"]

    flip_xz = fc.get("flip_xz_recordings", [])

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir} 中没有 csv 文件")

    print("=== 批量预处理：静态对齐 -> 低通 -> 降采样 ===")
    for csv_path in csv_files:
        out_path = out / f"{csv_path.stem}.npz"

        if out_path.exists():
            print(f"跳过（已存在） {csv_path.stem}")
            continue

        print(f"处理中: {csv_path.name}")
        proc = DataPreprocess(
            csv_path,
            preprocess_cfg=fc,
            cutoff=fc.get("lowpass_cutoff", 25.0),
            order=fc.get("lowpass_order", 4),
            flip_xz=flip_xz,
        )
        proc.preprocess()

        if proc.features is None or len(proc.features) == 0:
            print(f"  跳过（无有效数据） {csv_path.stem}")
            continue

        proc.save(out)

    print("全部预处理完成")


if __name__ == "__main__":
    preprocess_all(load_config())