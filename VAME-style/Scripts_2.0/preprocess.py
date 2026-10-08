from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

from data import Data
from utils import ROOT, load_config, output_dir


def lowpass_filter(data, cutoff, fs, order=4):
    """对多变量时序数据做零相位 Butterworth 低通滤波。"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


class DataPreprocess(Data):
    """IMU 预处理：对 raw 6 轴做低通滤波后保存。"""

    def __init__(self, file_path, preprocess_cfg, cutoff=25.0, order=4):
        super().__init__(file_path)
        self.cutoff = cutoff
        self.order = order
        self.features = None
        self.cfg_preproc = preprocess_cfg

    def preprocess(self):
        raw_6axis = np.hstack([self.acc, self.gyro])

        dt = np.median(np.diff(self.time))
        fs_orig = 1.0 / dt if dt > 0 else 100.0
        fs_target = self.cfg_preproc.get("target_fs", 25.0)

        # 抗混叠低通滤波：截止频率为目标的奈奎斯特频率
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

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{raw_dir} 中没有 csv 文件")

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
        )
        proc.preprocess()
        proc.save(out)
    print("全部预处理完成")


if __name__ == "__main__":
    preprocess_all(load_config())