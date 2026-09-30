# -*- coding: utf-8 -*-
"""探针 3：谱峰突出度 vs 自相关峰，分离噪声地板。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vamestyle.dataset import Session  # noqa: E402
from vamestyle.mixing import periodicity_trace, WIN, STEP  # noqa: E402


def prom_trace(x, fs=25.0):
    """逐帧谱峰突出度：滑窗 1.5 s，0.5–8 Hz 带内
    max(功率)/median(功率)。白噪声≈1，纯节律≫1。"""
    from numpy.lib.stride_tricks import sliding_window_view
    T = len(x)
    out = np.full(T, np.nan)
    if T < WIN + 10:
        return out
    x = x.astype(np.float64)
    k = np.ones(WIN) / WIN
    xd = x - np.convolve(x, k, "same")
    w = sliding_window_view(xd, WIN)
    idx = np.arange(0, len(w), STEP)
    ww = w[idx] * np.hanning(WIN)
    P = np.abs(np.fft.rfft(ww, axis=1)) ** 2
    fr = np.fft.rfftfreq(WIN, 1 / fs)
    band = (fr >= 0.5) & (fr <= 8.0)
    Pb = P[:, band]
    pr = Pb.max(1) / (np.median(Pb, axis=1) + 1e-12)
    for j, i in enumerate(idx):
        out[i:i + STEP] = pr[j]
    out[idx[-1]:min(idx[-1] + STEP + 30, T)] = pr[-1]
    return out


def q(r, tag):
    r = r[~np.isnan(r)]
    qq = np.percentile(r, [50, 75, 90, 99])
    print(f"  {tag:26s}: p50={qq[0]:5.1f} p75={qq[1]:5.1f} "
          f"p90={qq[2]:5.1f} p99={qq[3]:5.1f}")

# 合成地板检验
rng = np.random.default_rng(0)
t = np.arange(2500) / 25
sine = 0.6 * np.sin(2 * np.pi * 3 * t) + 0.4 * rng.standard_normal(2500)
noise = rng.standard_normal(2500)
burst = np.where((t % 3.0) < 0.4, 2.0, 0.1) * rng.standard_normal(2500)
print("[合成] 3Hz正弦 / 白噪声 / 突发噪声：")
q(prom_trace(sine), "正弦")
q(prom_trace(noise), "白噪声")
q(prom_trace(burst), "突发噪声")

s = Session("rec_000")
raw = s.raw6[:, 0:3].astype(np.float64)
print("\n[rec_000 raw acc] 谱峰突出度（3 轴 max）：")
ptr = np.nanmax(np.stack([prom_trace(raw[:, i]) for i in range(3)]), axis=0)
q(ptr, "raw max-axes")
print("\n[rec_000 raw acc] 自相关峰（对照）：")
atr = np.nanmax(np.stack([periodicity_trace(raw[:, i]) for i in range(3)]), axis=0)
q(atr, "raw max-axes")

# 两指标联合：散点分箱
pv, av = ptr, atr
ok = ~np.isnan(pv) & ~np.isnan(av)
pv, av = pv[ok], av[ok]
print("\n[prom 分箱内的 ac 均值]：")
for lo, hi in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]:
    m = (pv >= lo) & (pv < hi)
    if m.sum():
        print(f"  prom {lo:>4.0f}-{hi:<4.0f}: n={m.sum():>6} ac50={np.median(av[m]):.2f}")
np.save(Path(__file__).parent.parent / "cache" / "prom_rec000.npy", ptr)
