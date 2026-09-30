"""
IMU-Former 训练集定义模块
=========================
从预处理好的 CSV 文件加载 IMU 数据，切分为固定长度窗口供训练。

数据格式（与用户数据一致）：
  列: time, aGx, aGy, aGz, anGx, anGy, anGz, wx, wy, wz, interp_mask
  - aG*:  重力加速度 (3 轴)
  - anG*: 非重力加速度 (3 轴)
  - w*:   角速度 (3 轴, deg/s)
  - interp_mask: 1=插值位置, 0=原始数据

设计要点：
  1. 支持单文件或多文件载入（多段记录作为训练集）
  2. 固定 2s 窗口 (512 点 @ 256Hz)，50% 重叠滑窗
  3. 不做统计归一化（绝对幅值有行为意义）
  4. 可选固定单位缩放（如 gyro deg/s→rad/s），非统计操作
  5. interp_mask 随窗口一起返回，供后续重建损失加权使用
  6. 每段记录独立加载，window_index 记录 (文件序号, 起始采样点)
"""

import logging
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import IMUFormerConfig

logger = logging.getLogger(__name__)


def _np_to_tensor(arr: "np.ndarray") -> torch.Tensor:
    """
    numpy 数组 → torch 张量，但不经过 torch.from_numpy。

    原因：torch.from_numpy 需要 torch 在编译期链接的 numpy C-API 可用。
    当当前环境的 torch 构建版本与已安装的 numpy C-API 版本不匹配时
    （即便 `import numpy` / `import pandas` 在 Python 层都正常），
    torch 内部的 numpy 绑定会失败并抛出 "Numpy is not available"。
    frombuffer 直接读取原始字节缓冲，完全绕开 numpy C-API，任何环境都能工作。
    """
    return torch.frombuffer(
        arr.tobytes(), dtype=torch.float32, count=arr.size
    ).reshape(arr.shape).clone()


class IMUWindowDataset(Dataset):
    """
    PyTorch Dataset：IMU 窗口化数据集。

    Parameters
    ----------
    csv_paths : str | Path | list[str | Path]
        预处理好的 CSV 文件路径（单个或多个）。
    config : IMUFormerConfig
        配置对象（提供 window_len, hop_len, channel_names 等）。
    augment : bool, default False
        是否启用增广（最小验证阶段设为 False）。
    """

    def __init__(
        self,
        csv_paths: Union[str, Path, List[Union[str, Path]]],
        config: IMUFormerConfig,
        augment: bool = False,
    ):
        self.config = config
        self.augment = augment

        # 统一为列表
        if isinstance(csv_paths, (str, Path)):
            csv_paths = [csv_paths]
        self.csv_paths: List[Path] = [Path(p) for p in csv_paths]

        # 加载所有记录到内存
        self.recordings: List[Dict[str, Any]] = []
        self._load_all()

        # 构建窗口索引
        self.window_index: List[tuple] = []  # (rec_idx, start_sample)
        self._build_window_index()

        logger.info(
            "IMUWindowDataset 就绪: %d 段记录, %d 个窗口 "
            "(窗口=%d 点=%.1fs, 步进=%d 点=%.1fs)",
            len(self.recordings), len(self.window_index),
            config.window_len, config.window_sec,
            config.hop_len, config.hop_sec,
        )

    # ─────────────────────────────────────────────────────────────
    # 内部加载逻辑
    # ─────────────────────────────────────────────────────────────

    def _load_all(self):
        """逐文件加载 CSV，存入 self.recordings。"""
        col_names = list(self.config.channel_names)
        mask_col = self.config.mask_col

        for path in self.csv_paths:
            if not path.exists():
                raise FileNotFoundError(f"CSV 文件不存在: {path}")

            logger.info("加载 %s ...", path.name)

            # 只读取需要的列，指定 float32 节省内存
            use_cols = col_names + [mask_col]
            df = pd.read_csv(path, usecols=use_cols, dtype=np.float32)

            data = df[col_names].values  # (N, 9)
            mask = df[mask_col].values    # (N,)

            n_samples = len(data)
            duration_min = n_samples / self.config.sample_rate / 60

            # 基本检查
            if n_samples < self.config.window_len:
                logger.warning(
                    "%s: 仅 %d 点 (< %d 需要), 跳过",
                    path.name, n_samples, self.config.window_len,
                )
                continue

            # NaN/Inf 检查
            n_bad = int(np.isnan(data).sum() + np.isinf(data).sum())
            if n_bad > 0:
                logger.warning("%s: %d 个 NaN/Inf, 置零", path.name, n_bad)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            self.recordings.append({
                'data': data,          # (N, 9) float32
                'mask': mask,          # (N,) float32
                'name': path.stem,
                'n_samples': n_samples,
            })

            # 日志：时长 + 插值比例 + 各模态统计
            interp_ratio = float(mask.mean())
            logger.info(
                "  %s: %.1f min, %d 点, 插值占比 %.2f%%",
                path.stem, duration_min, n_samples, interp_ratio * 100,
            )
            for m_idx, m_name in enumerate(self.config.modality_names):
                ch = self.config.modality_groups[m_idx]
                vals = data[:, ch]
                logger.info(
                    "    [%s] mean=%+.4f std=%.4f min=%+.4f max=%+.4f",
                    m_name, vals.mean(), vals.std(), vals.min(), vals.max(),
                )

    def _build_window_index(self):
        """构建扁平索引: [(rec_idx, start), ...]。丢弃不完整的尾部。"""
        win_len = self.config.window_len
        hop = self.config.hop_len

        for rec_idx, rec in enumerate(self.recordings):
            n = rec['n_samples']
            n_windows = max(0, (n - win_len) // hop + 1)
            for w in range(n_windows):
                self.window_index.append((rec_idx, w * hop))

    # ─────────────────────────────────────────────────────────────
    # Dataset 接口
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec_idx, start = self.window_index[idx]
        rec = self.recordings[rec_idx]

        end = start + self.config.window_len
        window_data = rec['data'][start:end]     # (512, 9)
        window_mask = rec['mask'][start:end]      # (512,)

        # 可选：固定单位缩放（非统计归一化，对所有样本施加相同常数）
        if self.config.channel_scales is not None:
            scales = np.array(self.config.channel_scales, dtype=np.float32)
            window_data = window_data * scales[np.newaxis, :]

        # numpy → torch（不依赖 torch.from_numpy，绕开 numpy C-API 绑定问题）
        x = _np_to_tensor(window_data)                        # (512, 9)
        m = _np_to_tensor(window_mask.astype(np.float32))     # (512,)

        return {
            'x': x,               # (window_len, 9) 时间 × 通道
            'mask': m,            # (window_len,) 1=插值, 0=原始
            'file_idx': rec_idx,
            't_start': start,     # 起始采样点序号
        }

    # ─────────────────────────────────────────────────────────────
    # 辅助方法
    # ─────────────────────────────────────────────────────────────

    def get_recording_info(self) -> List[Dict]:
        """返回各记录的摘要信息。"""
        info = []
        for i, rec in enumerate(self.recordings):
            info.append({
                'index': i,
                'name': rec['name'],
                'n_samples': rec['n_samples'],
                'duration_min': rec['n_samples'] / self.config.sample_rate / 60,
                'interp_ratio': float(rec['mask'].mean()),
                'n_windows': sum(
                    1 for ri, _ in self.window_index if ri == i
                ),
            })
        return info


# ═══════════════════════════════════════════════════════════════
# 独立验证模块
# ═══════════════════════════════════════════════════════════════

def _generate_synthetic_csv(path: str, n_samples: int = 76800, sr: int = 256):
    """
    生成一个合成 CSV 文件用于验证（不依赖真实数据）。

    生成 5 分钟数据 (76800 点 @ 256Hz)，各通道用不同频率的正弦波 +
    少量随机噪声，模拟 IMU 信号。interp_mask 约 5% 随机置 1。
    """
    t = np.arange(n_samples) / sr
    rng = np.random.default_rng(42)

    # 重力：~1g，缓慢波动（模拟倾角变化）
    aGx = 0.3 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.01, n_samples)
    aGy = 0.5 * np.sin(2 * np.pi * 0.3 * t + 1.0) + rng.normal(0, 0.01, n_samples)
    aGz = -0.8 + 0.1 * np.sin(2 * np.pi * 0.2 * t) + rng.normal(0, 0.01, n_samples)

    # 非重力加速度：0.1-0.3g，中频波动（模拟运动）
    anGx = 0.15 * np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 0.02, n_samples)
    anGy = 0.10 * np.sin(2 * np.pi * 3.0 * t + 0.5) + rng.normal(0, 0.02, n_samples)
    anGz = 0.05 * np.sin(2 * np.pi * 1.5 * t + 1.2) + rng.normal(0, 0.02, n_samples)

    # 角速度：50-150 deg/s，高频（模拟头部转动）
    wx = 80 * np.sin(2 * np.pi * 5.0 * t) + rng.normal(0, 5, n_samples)
    wy = 60 * np.sin(2 * np.pi * 4.0 * t + 0.3) + rng.normal(0, 5, n_samples)
    wz = 100 * np.sin(2 * np.pi * 6.0 * t + 0.7) + rng.normal(0, 5, n_samples)

    # 插值掩码：~5% 随机
    interp_mask = rng.integers(0, 20, n_samples).astype(np.float32)  # ~5% 为非零
    interp_mask = (interp_mask > 18).astype(np.float32)

    df = pd.DataFrame({
        'time': t,
        'aGx': aGx.astype(np.float32), 'aGy': aGy.astype(np.float32), 'aGz': aGz.astype(np.float32),
        'anGx': anGx.astype(np.float32), 'anGy': anGy.astype(np.float32), 'anGz': anGz.astype(np.float32),
        'wx': wx.astype(np.float32), 'wy': wy.astype(np.float32), 'wz': wz.astype(np.float32),
        'interp_mask': interp_mask,
    })
    df.to_csv(path, index=False)
    return path


def _verify():
    """验证 Dataset 的正确性。"""
    import tempfile, os

    print("══════ IMUWindowDataset 验证 ══════\n")
    cfg = IMUFormerConfig()

    # --- 测试 1: 合成数据 ---
    print("[1] 生成合成 CSV (5 min @ 256Hz) ...")
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, 'synthetic_imu.csv')
    _generate_synthetic_csv(csv_path, n_samples=76800, sr=256)
    print(f"    文件: {csv_path}")
    print(f"    大小: {os.path.getsize(csv_path) / 1024 / 1024:.1f} MB")

    # --- 测试 2: 单文件加载 ---
    print("\n[2] 单文件加载 ...")
    ds = IMUWindowDataset(csv_path, cfg)
    print(f"    记录数: {len(ds.recordings)}")
    print(f"    窗口数: {len(ds)}")
    expected_windows = (76800 - 512) // 256 + 1  # = 298
    assert len(ds) == expected_windows, (
        f"窗口数应为 {expected_windows}, 得到 {len(ds)}"
    )
    print(f"    预期窗口数: {expected_windows} ✓")

    # --- 测试 3: 单窗口内容 ---
    print("\n[3] 单窗口内容检查 ...")
    sample = ds[0]
    x = sample['x']
    mask = sample['mask']
    assert x.shape == (512, 9), f"x.shape 应为 (512, 9), 得到 {x.shape}"
    assert mask.shape == (512,), f"mask.shape 应为 (512,), 得到 {mask.shape}"
    assert not torch.isnan(x).any(), "x 不应含 NaN"
    assert sample['file_idx'] == 0
    assert sample['t_start'] == 0
    print(f"    x: {x.shape} dtype={x.dtype} ✓")
    print(f"    mask: {mask.shape}, 插值占比={mask.mean():.3f}")
    print(f"    file_idx={sample['file_idx']}, t_start={sample['t_start']}")

    # 各模态值域检查
    for m_idx, m_name in enumerate(cfg.modality_names):
        ch = cfg.modality_groups[m_idx]
        vals = x[:, ch]
        print(f"    [{m_name}] min={vals.min():.4f} max={vals.max():.4f} "
              f"mean={vals.mean():.4f}")

    # --- 测试 4: 多文件加载 ---
    print("\n[4] 多文件加载 ...")
    csv_path2 = os.path.join(tmpdir, 'synthetic_imu_2.csv')
    _generate_synthetic_csv(csv_path2, n_samples=38400, sr=256)  # 2.5 min
    ds_multi = IMUWindowDataset([csv_path, csv_path2], cfg)
    print(f"    记录数: {len(ds_multi.recordings)}")
    print(f"    总窗口数: {len(ds_multi)}")
    assert len(ds_multi.recordings) == 2

    # 检查文件索引分布
    file_indices = [ds_multi[i]['file_idx'] for i in range(len(ds_multi))]
    n_file0 = sum(1 for fi in file_indices if fi == 0)
    n_file1 = sum(1 for fi in file_indices if fi == 1)
    print(f"    文件 0 窗口: {n_file0}, 文件 1 窗口: {n_file1}")
    assert n_file0 + n_file1 == len(ds_multi)

    # --- 测试 5: DataLoader 兼容性 ---
    print("\n[5] DataLoader 兼容性 ...")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"    batch['x']: {batch['x'].shape}")        # (8, 512, 9)
    print(f"    batch['mask']: {batch['mask'].shape}")    # (8, 512)
    assert batch['x'].shape == (8, 512, 9)
    assert batch['mask'].shape == (8, 512)
    print(f"    ✓ DataLoader 正常工作")

    # --- 测试 6: 记录信息 ---
    print("\n[6] 记录摘要 ...")
    for info in ds.get_recording_info():
        print(f"    #{info['index']} {info['name']}: "
              f"{info['duration_min']:.1f} min, {info['n_windows']} 窗口, "
              f"插值 {info['interp_ratio']*100:.1f}%")

    # 清理
    import shutil
    shutil.rmtree(tmpdir)

    print("\n[OK] IMUWindowDataset 验证全部通过")


if __name__ == '__main__':
    _verify()
