# -*- coding: utf-8 -*-
"""探针 2：max-over-axes 指标 + 100 Hz 对照。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vamestyle.dataset import Session  # noqa: E402
from vamestyle.mixing import periodicity_trace  # noqa: E402


def q(r, tag):
    r = r[~np.isnan(r)]
    qq = np.percentile(r, [50, 75, 90, 99])
    print(f"  {tag:26s}: r50={qq[0]:.2f} r75={qq[1]:.2f} "
          f"r90={qq[2]:.2f} r99={qq[3]:.2f}  (n={len(r)})")


s = Session("rec_000")
raw = s.raw6[:, 0:3].astype(np.float64)
tr = np.stack([periodicity_trace(raw[:, i]) for i in range(3)])
rmax = np.nanmax(tr, axis=0)
print("[25 Hz] max over 3 raw acc axes:")
q(rmax, "raw acc max-axes")

print("\n[100 Hz raw6] 裁 12000 后 ::4 抽取：")
d = np.load(r"F:\Kimi\IMU\VAME-IMU\data\rec_000.npz")
a = d["raw6"][12000:-12000, 0:3].astype(np.float64)
a25 = a[::4]
tr100 = np.stack([periodicity_trace(a25[:, i]) for i in range(3)])
q(np.nanmax(tr100, axis=0), "100Hz->25 max-axes")
q(periodicity_trace(a25[:, 0]), "100Hz->25 acc x")

# 强周期段实际占比：rmax 连续 >0.6 的段总长
m = rmax > 0.6
print(f"\n[25 Hz] rmax>0.6 帧占 {m.mean():.1%}，最长连续段 "
      f"{max((len(list(g)) for k, g in __import__('itertools').groupby(m) if k), default=0) / 25:.1f} s")
m99 = rmax > 0.8
print(f"[25 Hz] rmax>0.8 帧占 {m99.mean():.1%}")
