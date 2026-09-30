# -*- coding: utf-8 -*-
"""一次性探针：周期性到底在 raw acc 还是 linacc 里？

对 rec_000（25 Hz）比较同一 periodicity_trace 指标在
  (a) raw acc 单轴 x/y/z（含重力，用户肉眼看的通道）
  (b) linacc RMS（z 归一化后的包络，现 mixing 指标）
下的分位数。若 (a) 显著高于 (b) → 周期性的载体没进模型输入。
再对 100 Hz 原始 npz 重算 (a)，排除降采样因素。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vamestyle.dataset import CFG, Session  # noqa: E402
from vamestyle.mixing import periodicity_trace  # noqa: E402


def q(r, tag):
    r = r[~np.isnan(r)]
    if len(r) == 0:
        print(f"  {tag:24s}: 无有效窗")
        return
    qq = np.percentile(r, [50, 75, 90, 99])
    print(f"  {tag:24s}: r50={qq[0]:.2f} r75={qq[1]:.2f} "
          f"r90={qq[2]:.2f} r99={qq[3]:.2f}  (n={len(r)})")


def main():
    s = Session("rec_000")
    print(f"rec_000: T={s.T} @25Hz")

    print("\n[25 Hz npz]")
    raw = s.raw6[:, 0:3]                      # raw acc, 含重力
    for ax, nm in enumerate("xyz"):
        q(periodicity_trace(raw[:, ax]), f"raw acc {nm}")
    lin = s.feat9[:, 0:3]
    q(periodicity_trace(np.sqrt(np.mean(lin * lin, 1))), "linacc RMS (现指标)")
    for ax, nm in enumerate("xyz"):
        q(periodicity_trace(lin[:, ax]), f"linacc {nm} (z-norm)")
    gyr = s.feat9[:, 6:9]
    q(periodicity_trace(np.sqrt(np.mean(gyr * gyr, 1))), "gyro RMS (z-norm)")

    # 100 Hz 原始数据（未做动态分离/归一化/降采样）
    p100 = Path(r"F:\Kimi\IMU\VAME-IMU\data\rec_000.npz")
    if p100.exists():
        d = np.load(p100)
        print(f"\n[100 Hz 原始 npz] fields={d.files}")
        # 找 acc 字段：常见命名 acc / acc_g
        acc = None
        for k in ("acc", "acc_g", "raw_acc"):
            if k in d.files:
                acc = d[k]
                print(f"  使用字段 {k}, shape={acc.shape}")
                break
        if acc is None:
            for k in d.files:
                if d[k].ndim == 2 and d[k].shape[1] >= 3:
                    print(f"  候选字段 {k}: shape={d[k].shape}")
        else:
            # 裁首尾 12000 点与上游口径对齐，再降采样到 25 Hz 对齐时间轴
            a = acc[12000:-12000, 0:3].astype(np.float64)
            a25 = a[::4]                       # 100->25 简单抽取（只作对比）
            for ax, nm in enumerate("xyz"):
                q(periodicity_trace(a25[:, ax]), f"100Hz->25 acc {nm}")
    else:
        print(f"\n[100 Hz] 未找到 {p100}")


if __name__ == "__main__":
    main()
