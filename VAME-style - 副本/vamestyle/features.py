# -*- coding: utf-8 -*-
"""v4.0 手工特征旁路（12 维）+ 能量包络。

口径（对齐架构图通道 B 与 params：band_n=8, band_fmax=12 Hz, win=75=3 s @25 Hz）：
  hand12 = [log能量(1) | 频带能量(8) | grav姿态(3)]
  - log 能量：动态 6 通道（linacc3+gyro3，**不含 grav**——v2.0 决策 5）
    窗内 rms² 均值取 log10；兼作全部验收测试的标准答案（能量包络）
  - 频带能量：窗内 dyn6 平均 PSD 在 0–12 Hz 均分 8 带的 log 带功率
  - grav 姿态：窗内 grav3（单位向量，原尺度）均值
  每帧取以该帧为中心的对称窗（与 VAME 窗中心约定一致）；
  输出截断到 [15 : T-14]（长 T-29），与 mu latent 行对齐。
  标准化：逐 session robust-z（median/IQR，跨 session 可比）。

幂等缓存（cache/）：
  energy_<session>.npy  (T-29,)   原始 log 能量包络（未标准化，验收标准答案）
  hand_<session>.npy    (T-29,12) 未标准化手工特征
  zfeat_<session>.npy   (T-29,28) [mu(16) ‖ hand12 robust-z]
  hand_c_<session>.npy  (T-29,13) 路线 C：[logE(1)|band(8)|θ|sinφ|cosφ|grav_std]
  zfeat_c_<session>.npy (T-29,29) [mu(16) ‖ hand13 robust-z]

用法：python -m vamestyle.features            # 旧 12 维口径（幂等）
      python -m vamestyle.features c          # 路线 C 13 维口径（幂等）
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from vamestyle.dataset import CFG, ROOT, sessions

FEAT = CFG["features"]
WIN = int(FEAT["win"])                 # 75 = 3 s @25 Hz
BAND_N = int(FEAT["band_n"])           # 8
FMAX = float(FEAT["band_fmax"])        # 12 Hz
FS = float(CFG["data"]["fs"])
HALF = WIN // 2
EPS = 1e-12

# 8 个频带的边界（线性，0–12 Hz，含奈奎斯特）
BAND_EDGES = np.linspace(0.0, FMAX, BAND_N + 1)
# rfft 频率轴（75 点 @25 Hz → 0..12 Hz，分辨率 1/3 Hz）
_FFT_FREQS = np.fft.rfftfreq(WIN, d=1.0 / FS)


def _robust_z(x):
    """(N,D) 逐列 median/IQR 标准化；IQR≈0 的列保持 0。"""
    med = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    z = np.where(iqr > 1e-8, (x - med) / np.where(iqr > 1e-8, iqr, 1.0), 0.0)
    return z.astype(np.float32)


def hand_features(sess):
    """单 session → (energy (T,), hand (T,12))，全帧（未截断、未标准化）。"""
    dyn = sess.dyn6.astype(np.float32)          # (T,6)
    grav = sess.grav.astype(np.float32)          # (T,3)
    T = sess.T
    pad = np.pad(dyn, ((HALF, HALF), (0, 0)), mode="edge")
    w = sliding_window_view(pad, WIN, axis=0)    # (T,6,WIN)

    # log 能量（dyn6 rms² 均值 → log10）
    energy = np.log10(np.mean(np.mean(w * w, axis=2), axis=1) + EPS)

    # 频带能量：dyn6 平均 PSD → 8 带 log 带功率
    psd = np.abs(np.fft.rfft(w, axis=2)) ** 2 / WIN   # (T,6,F)
    psd = psd.mean(axis=1)                             # (T,F)
    bands = []
    for i in range(BAND_N):
        # 最后一带含右边界（纳入奈奎斯特 12 Hz 分量）
        if i < BAND_N - 1:
            m = (_FFT_FREQS >= BAND_EDGES[i]) & (_FFT_FREQS < BAND_EDGES[i + 1])
        else:
            m = (_FFT_FREQS >= BAND_EDGES[i]) & (_FFT_FREQS <= BAND_EDGES[i + 1])
        bands.append(np.log10(psd[:, m].mean(axis=1) + EPS))
    band_feat = np.stack(bands, axis=1)                # (T,8)

    # grav 姿态：窗内均值（原尺度单位向量）
    pg = np.pad(grav, ((HALF, HALF), (0, 0)), mode="edge")
    wg = sliding_window_view(pg, WIN, axis=0)
    grav_mean = wg.mean(axis=2)                         # (T,3)

    hand = np.concatenate([energy[:, None], band_feat, grav_mean], axis=1)
    return energy.astype(np.float32), hand.astype(np.float32)


def hand_features_c(sess, win=None):
    """路线 C 旁路（13 维）：hand13 = [logE(1) | band(8) | θ(1) | sinφ(1)
    | cosφ(1) | grav_std(1)]。

    与 hand_features 的差异只在姿态 3 列 → 4 列：
      - 姿态在逐鼠垂直零点系计算（R_s = 本 session 重力中位 → (0,0,-1)
        的最小弧旋转，复用 posture._align_rotation），与姿态流形图同系；
      - θ = 窗内旋转 grav 均值的倾角（°，[0,180]）；
      - φ = 倾斜方位，以 sin/cos 两维编码（环形变量，消除 ±180° 接缝）；
      - grav_std = 窗内逐帧 grav 差分 L2 均值（姿态稳定性）。
    logE/频带 9 列与 hand_features 逐位一致。

    win：滑窗长度（帧），默认 WIN=75；win=31 时窗中心 ±15 帧，与 VAME
    输入窗（time_window=30）完全同尺度（v4.1 对照实验，2026-09-29）。"""
    from vamestyle.posture import _align_rotation
    win = int(win or WIN)
    half = win // 2
    band_edges = np.linspace(0.0, FMAX, BAND_N + 1)
    fft_freqs = np.fft.rfftfreq(win, d=1.0 / FS)
    dyn = sess.dyn6.astype(np.float32)          # (T,6)
    grav = sess.grav.astype(np.float64)          # (T,3)
    R = _align_rotation(np.median(grav, axis=0))
    grav_r = grav @ R.T                          # 逐鼠垂直零点系
    T = sess.T
    pad = np.pad(dyn, ((half, half), (0, 0)), mode="edge")
    w = sliding_window_view(pad, win, axis=0)    # (T,6,win)

    # log 能量（与 hand_features 同式）
    energy = np.log10(np.mean(np.mean(w * w, axis=2), axis=1) + EPS)

    # 频带能量（与 hand_features 同式）
    psd = np.abs(np.fft.rfft(w, axis=2)) ** 2 / win
    psd = psd.mean(axis=1)
    bands = []
    for i in range(BAND_N):
        if i < BAND_N - 1:
            m = (fft_freqs >= band_edges[i]) & (fft_freqs < band_edges[i + 1])
        else:
            m = (fft_freqs >= band_edges[i]) & (fft_freqs <= band_edges[i + 1])
        bands.append(np.log10(psd[:, m].mean(axis=1) + EPS))
    band_feat = np.stack(bands, axis=1)          # (T,8)

    # 姿态 4 列（旋转系）
    pg = np.pad(grav_r, ((half, half), (0, 0)), mode="edge")
    wg = sliding_window_view(pg, win, axis=0)    # (T,3,win)
    gm = wg.mean(axis=2)                          # (T,3) 窗内均值
    gm_u = gm / np.maximum(np.linalg.norm(gm, axis=1, keepdims=True), 1e-12)
    theta = np.degrees(np.arccos(np.clip(-gm_u[:, 2], -1, 1)))    # (T,)
    r_xy = np.hypot(gm[:, 0], gm[:, 1])
    sin_phi = np.where(r_xy > 1e-12, gm[:, 1] / np.maximum(r_xy, 1e-12), 0.0)
    cos_phi = np.where(r_xy > 1e-12, gm[:, 0] / np.maximum(r_xy, 1e-12), 0.0)
    dg = np.linalg.norm(np.diff(grav_r, axis=0), axis=1)          # (T-1,)
    dg_pad = np.pad(dg, (half, half + 1), mode="edge")            # (T+2H,)
    grav_std = sliding_window_view(dg_pad, win, axis=0).mean(axis=1)  # (T,)

    hand = np.concatenate([energy[:, None], band_feat,
                           theta[:, None], sin_phi[:, None],
                           cos_phi[:, None], grav_std[:, None]], axis=1)
    return energy.astype(np.float32), hand.astype(np.float32)


def build_c(names=None, win=None):
    """路线 C 全量构建（幂等）。返回 {name: zfeat_c (T-29,29)}。

    win=None：默认 WIN=75，缓存 zfeat_c_<sess>.npy（旧口径，不动）；
    win=31：v4.1 对照口径，缓存 zfeat_c_w31_<sess>.npy（窗中心 ±15 帧，
    与 VAME 输入窗同尺度）。"""
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    tag = "c" if win is None else f"c_w{int(win)}"
    zf = {}
    for s in sessions(names):
        p = out / f"zfeat_{tag}_{s.name}.npy"
        if p.exists():
            zf[s.name] = np.load(p)
            print(f"[feat_{tag}] {s.name}: 命中缓存 {zf[s.name].shape}")
            continue
        mu = np.load(out / f"mu_{s.name}.npy")           # (T-29,16)
        energy, hand = hand_features_c(s, win=win)       # (T,) (T,13)
        C = CFG["vame"]["time_window"] // 2
        energy = energy[C:s.T - C + 1]
        hand = hand[C:s.T - C + 1]
        assert len(hand) == len(mu), (len(hand), len(mu))
        hand_z = _robust_z(hand) if FEAT["robust_z"] else hand
        z = np.concatenate([mu, hand_z], axis=1).astype(np.float32)
        np.save(out / f"hand_{tag}_{s.name}.npy", hand)
        np.save(p, z)
        zf[s.name] = z
        print(f"[feat_{tag}] {s.name}: hand{hand.shape} zfeat{z.shape}")
    return zf


def build(names=None):
    """全量构建（幂等）。返回 {name: zfeat (T-29,28)}。"""
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    zf = {}
    for s in sessions(names):
        p = out / f"zfeat_{s.name}.npy"
        if p.exists():
            zf[s.name] = np.load(p)
            print(f"[feat] {s.name}: 命中缓存 {zf[s.name].shape}")
            continue
        mu = np.load(out / f"mu_{s.name}.npy")           # (T-29,16)
        energy, hand = hand_features(s)                  # (T,) (T,12)
        # mu 行 i ↔ 原始帧 i+15（VAME 30 帧窗中心）→ 取帧 15..T-15（含），长 T-29
        C = CFG["vame"]["time_window"] // 2
        energy = energy[C:s.T - C + 1]
        hand = hand[C:s.T - C + 1]
        assert len(hand) == len(mu), (len(hand), len(mu))
        hand_z = _robust_z(hand) if FEAT["robust_z"] else hand
        z = np.concatenate([mu, hand_z], axis=1).astype(np.float32)
        np.save(out / f"energy_{s.name}.npy", energy)
        np.save(out / f"hand_{s.name}.npy", hand)
        np.save(p, z)
        zf[s.name] = z
        print(f"[feat] {s.name}: energy{energy.shape} hand{hand.shape} zfeat{z.shape}")
    return zf


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "c":
        zf = build_c()
    else:
        zf = build()
    z = np.concatenate(list(zf.values()), axis=0)
    print(f"[feat] 合计 {z.shape}；NaN={int(np.isnan(z).sum())}")
