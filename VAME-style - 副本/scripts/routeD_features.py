# -*- coding: utf-8 -*-
"""路线 D 旁路特征（19 维 @50 Hz）+ zfeat_D 组装。

hand19 = [logE(1) | band8(0–12 Hz log 带功率) | rms6(逐轴窗内 RMS) |
          phase4(R_dom 节律一致性, sinψ, cosψ, f* 主导频率)]
  - 窗 win=31（0.62 s，中心 ±15 帧，与 mu 行 i ↔ 帧 i+15 对齐），截断 [15, T-14]
  - 相位：dyn6 带通 1–12 Hz（butter4 零相位）→ Hilbert → 逐轴单位解析信号
    u_a(t)；窗内 u_a 复均值 → R_a=|·|（1=强节律），取带功率主导轴 a* 的
    (R_a*, ψ_a*)，ψ 用 sin/cos 编码（环形无缝缝）；f* = 主导 FFT bin 频率
  - 逐 session robust-z（median/IQR）
zfeat_D = [mu(16) ‖ hand19_z]（35 列）→ runs/routeD_posture_vae_50hz/outputs/
用法：venv_python scripts/routeD_features.py
"""
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, hilbert

WORKSPACE = Path(r"F:\Kimi\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
RUN = VAME_STYLE / "runs" / "routeD_posture_vae_50hz"
DS50 = WORKSPACE / "VAME-IMU" / "data" / "ds50"
MU_DIR = RUN / "outputs" / "mu"
OUT = RUN / "outputs"
SESS = ["rec_000", "rec_005", "rec_010"]

WIN = 31
HALF = WIN // 2
FS = 50.0
BAND_N = 8
FMAX = 12.0
BP_LO, BP_HI = 1.0, 12.0
EPS = 1e-12
BAND_EDGES = np.linspace(0.0, FMAX, BAND_N + 1)
FFT_FREQS = np.fft.rfftfreq(WIN, d=1.0 / FS)
C = 15                     # mu 行 i ↔ 帧 i+15 → 截断 [15, T-14]


def robust_z(x):
    med = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    return np.where(iqr > 1e-8, (x - med) / np.where(iqr > 1e-8, iqr, 1.0),
                    0.0).astype(np.float32)


def hand19(dyn):
    """dyn (T,6) 物理量纲 → (T-29, 19)（已截断对齐 mu）。"""
    T = len(dyn)
    pad = np.pad(dyn, ((HALF, HALF), (0, 0)), mode="edge")
    w = sliding_window_view(pad, WIN, axis=0)              # (T,6,31)

    logE = np.log10(np.mean(np.mean(w * w, axis=2), axis=1) + EPS)      # (T,)
    rms = np.sqrt(np.mean(w * w, axis=2) + EPS)                          # (T,6)

    psd = np.abs(np.fft.rfft(w, axis=2)) ** 2 / WIN                      # (T,6,16)
    psd_ax = psd.sum(axis=2)                                             # (T,6)
    bands = []
    for i in range(BAND_N):
        if i < BAND_N - 1:
            m = (FFT_FREQS >= BAND_EDGES[i]) & (FFT_FREQS < BAND_EDGES[i + 1])
        else:
            m = (FFT_FREQS >= BAND_EDGES[i]) & (FFT_FREQS <= BAND_EDGES[i + 1])
        bands.append(np.log10(psd[:, :, m].mean(axis=2) + EPS))          # (T,6)
    band8 = np.stack(bands, axis=1).mean(axis=2)                         # (T,8)

    from scipy.signal import sosfiltfilt
    sos = butter(4, [BP_LO / (FS / 2), BP_HI / (FS / 2)], btype="band",
                 output="sos")
    u = np.empty((T, 6), dtype=np.complex64)
    for a in range(6):
        bp = sosfiltfilt(sos, dyn[:, a].astype(np.float64))
        an = hilbert(bp)
        u[:, a] = (an / np.maximum(np.abs(an), 1e-12)).astype(np.complex64)
    ue = sliding_window_view(
        np.pad(u.real, ((HALF, HALF), (0, 0)), mode="edge"), WIN, axis=0)
    ui = sliding_window_view(
        np.pad(u.imag, ((HALF, HALF), (0, 0)), mode="edge"), WIN, axis=0)
    E = ue.mean(axis=2) + 1j * ui.mean(axis=2)                           # (T,6)
    R_a = np.abs(E)
    a_star = psd_ax[:, 1:].argmax(axis=1) + 1                            # (T,) 轴 0-5
    t_idx = np.arange(T)
    R = R_a[t_idx, a_star]
    psi = np.angle(E[t_idx, a_star])
    # 主导频率：主导轴的最强非零 bin
    psd_dom = psd[t_idx, a_star]                                         # (T,16)
    f_star = FFT_FREQS[psd_dom[:, 1:].argmax(axis=1) + 1]
    phase4 = np.stack([R, np.sin(psi), np.cos(psi), f_star], axis=1)

    hand = np.concatenate([logE[:, None], band8, rms, phase4], axis=1)   # (T,19)
    return hand[C: T - C + 1].astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SESS:
        d = np.load(DS50 / f"{s}.npz")
        dyn = d["raw6"].astype(np.float32)
        mu = np.load(MU_DIR / f"mu_{s}.npy")
        h = hand19(dyn)
        assert len(h) == len(mu), (s, len(h), len(mu))
        z = np.concatenate([mu, robust_z(h)], axis=1).astype(np.float32)
        np.save(OUT / f"hand19_{s}.npy", h)
        np.save(OUT / f"zfeat_D_{s}.npy", z)
        print(f"[D] {s}: mu{mu.shape} hand{h.shape} zfeat_D{z.shape}",
              flush=True)
    print("[D] 完成 →", OUT)


if __name__ == "__main__":
    main()
