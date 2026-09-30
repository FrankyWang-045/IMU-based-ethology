"""
IMU-Former 数据层统合验证
==========================
将 Dataset → DataLoader → Patcher 串联起来，端到端验证数据管线。

运行方式：
  cd src
  python -m imuformer.verify_data                    # 合成数据测试
  python -m imuformer.verify_data --csv path/to/file.csv  # 真实数据测试
"""

import sys
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

from .config import IMUFormerConfig
from .dataset import IMUWindowDataset, _generate_synthetic_csv
from .patching import ModalityPatcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_with_synthetic():
    """使用合成数据验证完整管线。"""
    print("══════════════════════════════════════════════════════")
    print("  IMU-Former 数据层统合验证（合成数据）")
    print("══════════════════════════════════════════════════════\n")

    cfg = IMUFormerConfig()
    print(cfg.summary())
    print()

    # ── 1. 生成合成数据 ──
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    csv1 = os.path.join(tmpdir, 'mouse1.csv')
    csv2 = os.path.join(tmpdir, 'mouse2.csv')
    _generate_synthetic_csv(csv1, n_samples=76800, sr=256)   # 5 min
    _generate_synthetic_csv(csv2, n_samples=38400, sr=256)   # 2.5 min
    print(f"[1] 合成数据: {csv1} (5min), {csv2} (2.5min)\n")

    # ── 2. 构建 Dataset ──
    print("[2] 构建 Dataset ...")
    ds = IMUWindowDataset([csv1, csv2], cfg)
    print(f"    总窗口数: {len(ds)}")
    for info in ds.get_recording_info():
        print(f"    #{info['index']} {info['name']}: "
              f"{info['duration_min']:.1f} min → {info['n_windows']} 窗口, "
              f"插值 {info['interp_ratio']*100:.1f}%")
    print()

    # ── 3. DataLoader ──
    print("[3] 构建 DataLoader (batch=16, shuffle=True) ...")
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=False)
    print(f"    批次数: {len(loader)}")
    print()

    # ── 4. Patching ──
    print("[4] 构建 ModalityPatcher ...")
    patcher = ModalityPatcher(cfg)
    print()

    # ── 5. 端到端：一个 batch 的完整流程 ──
    print("[5] 端到端测试：取一个 batch 走完整管线 ...")
    batch = next(iter(loader))
    x = batch['x']           # (16, 512, 9)
    mask = batch['mask']     # (16, 512)

    print(f"    原始 batch:  x={x.shape}, mask={mask.shape}")
    print(f"    x 值域: [{x.min():.4f}, {x.max():.4f}]")
    print(f"    mask 均值: {mask.mean():.4f} (插值占比)")

    # Patching
    patches, mod_ids, positions = patcher(x)
    print(f"    → patches: {patches.shape}")       # (16, 48, 192)
    print(f"    → mod_ids: {mod_ids.shape}")        # (16, 48)
    print(f"    → positions: {positions.shape}")     # (16, 48)

    # 验证形状
    assert patches.shape == (16, cfg.total_tokens, cfg.patch_flat_dim)
    assert mod_ids.shape == (16, cfg.total_tokens)
    assert positions.shape == (16, cfg.total_tokens)
    print(f"    形状检查 ✓\n")

    # ── 6. 往返重建 ──
    print("[6] 往返重建（patches → signal）...")
    x_recon = patcher.patches_to_signal(patches)
    max_err = (x - x_recon).abs().max().item()
    mean_err = (x - x_recon).abs().mean().item()
    print(f"    重建 max error: {max_err:.8f}")
    print(f"    重建 mean error: {mean_err:.8f}")
    assert max_err < 1e-5, f"重建误差过大: {max_err}"
    print(f"    往返一致 ✓\n")

    # ── 7. 各模态统计 ──
    print("[7] batch 内各模态值域统计 ...")
    for m_idx, m_name in enumerate(cfg.modality_names):
        ch = cfg.modality_groups[m_idx]
        vals = x[:, :, ch]  # (B, L, 3)
        print(f"    [{m_name:8s}] mean={vals.mean():+.4f} std={vals.std():.4f} "
              f"min={vals.min():+.4f} max={vals.max():+.4f}")
    print()

    # ── 8. 内存占用估算 ──
    print("[8] 内存占用估算 ...")
    # 数据在内存中：每段记录 (N, 9) float32 + (N,) mask float32 = 40 bytes/样本
    total_samples = sum(r['n_samples'] for r in ds.recordings)
    mem_mb = total_samples * 40 / 1024 / 1024
    print(f"    总采样点: {total_samples:,}")
    print(f"    内存占用: ~{mem_mb:.1f} MB (float32)")
    # 窗口数 × 窗口大小 × 通道 × float32
    win_mem = len(ds) * 512 * 9 * 4 / 1024 / 1024
    print(f"    若全展开为窗口: ~{win_mem:.1f} MB")
    print()

    # ── 9. 单段数据快速训练模拟 ──
    print("[9] 模拟训练前向传播（10 个 batch）...")
    import time
    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        x = batch['x']
        patches, _, _ = patcher(x)
        # 模拟后续：线性投影 + 编码器 + VQ + 解码器（此处只走 patching）
    elapsed = time.time() - t0
    n_processed = min(10, len(loader)) * 16
    print(f"    处理 {n_processed} 个窗口, 耗时 {elapsed:.3f}s")
    print(f"    吞吐量: {n_processed / elapsed:.0f} windows/s")
    print()

    # 清理
    import shutil
    shutil.rmtree(tmpdir)

    print("══════════════════════════════════════════════════════")
    print("  [OK] 数据层统合验证全部通过")
    print("══════════════════════════════════════════════════════")


def verify_with_real_csv(csv_path: str):
    """使用真实 CSV 数据验证。"""
    print("══════════════════════════════════════════════════════")
    print(f"  IMU-Former 数据层验证（真实数据: {Path(csv_path).name}）")
    print("══════════════════════════════════════════════════════\n")

    cfg = IMUFormerConfig()
    print(cfg.summary())
    print()

    # 加载
    ds = IMUWindowDataset(csv_path, cfg)
    print(f"\n窗口数: {len(ds)}")
    for info in ds.get_recording_info():
        print(f"  #{info['index']} {info['name']}: "
              f"{info['duration_min']:.1f} min → {info['n_windows']} 窗口, "
              f"插值 {info['interp_ratio']*100:.1f}%")

    # 取一个窗口检查
    sample = ds[0]
    x = sample['x']
    print(f"\n窗口 0: x={x.shape}, mask 插值占比={sample['mask'].mean():.4f}")

    # 各模态统计
    print("\n各模态值域:")
    for m_idx, m_name in enumerate(cfg.modality_names):
        ch = cfg.modality_groups[m_idx]
        vals = x[:, ch]
        print(f"  [{m_name:8s}] mean={vals.mean():+.4f} std={vals.std():.4f} "
              f"min={vals.min():+.4f} max={vals.max():+.4f}")

    # Patching
    patcher = ModalityPatcher(cfg)
    patches, mod_ids, pos = patcher(x.unsqueeze(0))  # 加 batch 维
    print(f"\nPatching: {patches.shape} → {cfg.total_tokens} tokens × {cfg.patch_flat_dim} dim")

    # 往返
    x_recon = patcher.patches_to_signal(patches)
    err = (x.unsqueeze(0) - x_recon).abs().max().item()
    print(f"往返误差: {err:.10f}")
    assert err < 1e-5, f"往返误差过大: {err}"

    # DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    batch = next(iter(loader))
    patches_b, _, _ = patcher(batch['x'])
    print(f"\nDataLoader batch: x={batch['x'].shape} → patches={patches_b.shape}")

    print("\n══════════════════════════════════════════════════════")
    print("  [OK] 真实数据验证通过")
    print("══════════════════════════════════════════════════════")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU-Former 数据层验证')
    parser.add_argument('--csv', type=str, default=None,
                        help='真实 CSV 文件路径（不提供则用合成数据）')
    args = parser.parse_args()

    if args.csv:
        verify_with_real_csv(args.csv)
    else:
        verify_with_synthetic()
