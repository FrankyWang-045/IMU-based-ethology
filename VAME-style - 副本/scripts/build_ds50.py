# -*- coding: utf-8 -*-
"""构建 50 Hz 预处理数据 ds50/（路线 D 及后续高频分析用）。

来源：VAME-IMU/data/*.npz（100 Hz 生产版，t 原点 120.00s）。
处理：butter4 低通 @25 Hz（= 新奈奎斯特，零相位 filtfilt）→ 2 倍抽取。
  - raw6：滤波后抽取（物理量纲 g / dps）
  - feat9：滤波后抽取；grav 三列（3:6）重归一化为单位向量
  - t：t[::2]（原点 120.00 s 不变，dt=0.02）
  - interp_mask：any-pooling 等价（直接 [::2]）
校验：ds50[::2] 的 raw6 与 ds25 相关系数（0–12 Hz 内容应≈一致，>0.995）。
用法：python build_ds50.py [rec_000,rec_001,...]（默认全部 14 条）
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

SRC = Path(r"F:\Kimi\IMU\VAME-IMU\data")
DST = SRC / "ds50"
FC = 25.0          # 低通截止 = 50 Hz 奈奎斯特
SRC_FS = 100.0


def decimate50(x, axis=0):
    b, a = butter(4, FC / (SRC_FS / 2))
    y = filtfilt(b, a, x, axis=axis)
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, None, 2)
    return y[tuple(sl)]


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else [
        f"rec_{i:03d}" for i in range(14)]
    DST.mkdir(parents=True, exist_ok=True)
    for n in names:
        d = np.load(SRC / f"{n}.npz")
        raw6 = decimate50(d["raw6"].astype(np.float64)).astype(np.float32)
        feat9 = decimate50(d["feat9"].astype(np.float64)).astype(np.float32)
        g = feat9[:, 3:6]
        g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        feat9[:, 3:6] = g.astype(np.float32)
        t = d["t"][::2]
        im = d["interp_mask"][::2]
        assert abs(t[1] - t[0] - 0.02) < 1e-9 and abs(t[0] - 120.0) < 1e-9
        assert len(raw6) == len(t)
        np.savez_compressed(DST / f"{n}.npz", raw6=raw6, feat9=feat9, t=t,
                            interp_mask=im)
        # 与 ds25 一致性校验（0-12 Hz 内容）
        d25 = np.load(SRC / "ds25" / f"{n}.npz")
        m = min(len(raw6) // 2, len(d25["raw6"]))
        corr = np.corrcoef(raw6[: 2 * m: 2, 0], d25["raw6"][:m, 0])[0, 1]
        print(f"[ds50] {n}: T={len(t)} t0={t[0]:.2f} "
              f"vs ds25 corr={corr:.5f}", flush=True)


if __name__ == "__main__":
    main()
