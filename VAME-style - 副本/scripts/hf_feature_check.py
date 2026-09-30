# -*- coding: utf-8 -*-
"""高频旁路候选特征 · 数据层检查（探针第 1 步，不训练任何模型）。

候选（8 列，全部从 ds25 npz 的 dyn6 直接计算）：
  hir_<ch>×6  分通道 5–12.5 Hz 能量占比（3 s 窗，与现有 band 同窗长）
  hi_short     短窗(0.6 s) 5–12.5 Hz log 能量（dyn6 通道均值）
  hi_entropy   3 s 窗 5–12.5 Hz 谱熵（dyn6 通道均值 PSD）

检查项：
  1) 冗余度：候选列 vs 现有 zfeat 28 列的最大 |Pearson r|
  2) 判别力：36 状态标签下的类间/总方差比 η²（与现有 28 列的 η² 分布对比）
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(r"F:\Kimi\IMU\VAME-Style")
NPZ_DIR = Path(r"F:\Kimi\IMU\VAME-IMU\data\ds25")
FS = 25.0
V12 = [f"rec_{i:03d}" for i in list(range(11)) + [12]]  # 排除 011/013
CH = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
WIN = 75      # 3 s
WIN2 = 15     # 0.6 s
OFF = 15      # zfeat 行 i ↔ 原始帧 i+15


def robust_z(x):
    med = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    return np.where(iqr > 1e-8, (x - med) / np.where(iqr > 1e-8, iqr, 1.0), 0.0)


def cand_features(dyn):
    """dyn (T,6) → (T,8) 候选特征（未标准化）。"""
    T = dyn.shape[0]
    freqs = np.fft.rfftfreq(WIN, d=1 / FS)
    hi = freqs >= 5.0

    pad = np.pad(dyn, ((WIN // 2, WIN // 2), (0, 0)), mode="edge")
    w = sliding_window_view(pad, WIN, axis=0)              # (T,6,75)
    psd = np.abs(np.fft.rfft(w, axis=2)) ** 2 / WIN        # (T,6,F)
    hi_sum = psd[:, :, hi].sum(axis=2)                     # (T,6)
    tot_sum = psd.sum(axis=2) + 1e-12
    hir = hi_sum / tot_sum                                 # 分通道高频比 (T,6)

    psd_m = psd.mean(axis=1)                               # (T,F) 通道均值
    p_hi = psd_m[:, hi]
    p = p_hi / (p_hi.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(hi.sum())  # 归一化谱熵

    pad2 = np.pad(dyn, ((WIN2 // 2, WIN2 // 2), (0, 0)), mode="edge")
    w2 = sliding_window_view(pad2, WIN2, axis=0)
    psd2 = (np.abs(np.fft.rfft(w2, axis=2)) ** 2 / WIN2).mean(axis=1)
    f2 = np.fft.rfftfreq(WIN2, d=1 / FS)
    short = np.log10(psd2[:, f2 >= 5.0].mean(axis=1) + 1e-12)

    return np.concatenate([hir, short[:, None], ent[:, None]], axis=1)


CAND_NAMES = [f"hir_{c}" for c in CH] + ["hi_short", "hi_entropy"]
cand_all, zfeat_all, lab_all = [], [], []
for s in V12:
    d = np.load(NPZ_DIR / f"{s}.npz")
    zf = np.load(ROOT / "cache" / f"zfeat_{s}.npy")
    lab = np.load(ROOT / "results" / s / "labels_k36_v12.npy")
    n = len(zf)
    cf = cand_features(d["raw6"].astype(np.float32))[OFF:OFF + n]
    assert len(cf) == n == len(lab), (s, len(cf), n, len(lab))
    cand_all.append(robust_z(cf))
    zfeat_all.append(zf)
    lab_all.append(lab)
    print(f"{s}: n={n}")

C = np.concatenate(cand_all)      # (N,8)  robust-z
Z = np.concatenate(zfeat_all)     # (N,28)
L = np.concatenate(lab_all)       # (N,)
N, Dc, Dz = len(C), C.shape[1], Z.shape[1]
print(f"\n总样本 N={N}, 候选 {Dc} 列, 现有 zfeat {Dz} 列, 状态数 {len(np.unique(L))}")


def eta2(x, g):
    """相关比 η² = 组间方差 / 总方差。"""
    gm = x.mean()
    ss_tot = ((x - gm) ** 2).sum()
    ss_b = 0.0
    for u in np.unique(g):
        m = g == u
        ss_b += m.sum() * (x[m].mean() - gm) ** 2
    return ss_b / ss_tot


# ---- 1) 冗余度 ----
Cz = (C - C.mean(0)) / (C.std(0) + 1e-12)
Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
redund = {}
for i, nm in enumerate(CAND_NAMES):
    r = (Cz[:, i, None] * Zz).mean(axis=0)   # 与 28 列的 Pearson r
    top = np.argsort(-np.abs(r))[:3]
    redund[nm] = dict(max_abs_r=round(float(np.abs(r).max()), 3),
                      top={f"col{int(j)}": round(float(r[j]), 3) for j in top})

# ---- 2) 判别力 η² ----
eta_cand = {nm: round(float(eta2(C[:, i], L)), 4) for i, nm in enumerate(CAND_NAMES)}
eta_exist = np.array([eta2(Z[:, j], L) for j in range(Dz)])

# ---- 图 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
names = CAND_NAMES
maxr = [redund[n]["max_abs_r"] for n in names]
ax.bar(range(Dc), maxr, color="#4C7FB0")
ax.set_xticks(range(Dc), names, rotation=35, ha="right", fontsize=8)
ax.axhline(0.9, color="r", ls="--", lw=0.8)
ax.set_title("冗余度：候选列 vs 现有 zfeat 28 列的最大 |r|")
ax.set_ylabel("max |Pearson r|"); ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(eta_exist, bins=20, alpha=0.6, color="#9AA0A6", label="现有 zfeat 28 列")
for i, nm in enumerate(names):
    ax.axvline(eta_cand[nm], color=plt.cm.tab10(i), lw=1.5, ls="--")
    ax.text(eta_cand[nm], 0.9 - i * 0.06, f"{nm}={eta_cand[nm]:.3f}",
            transform=ax.get_xaxis_transform(), rotation=0, fontsize=7,
            color=plt.cm.tab10(i))
ax.set_title("判别力：36 状态下 η²（类间/总方差比）")
ax.set_xlabel("η²"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

out_dir = ROOT / "outputs"
out_dir.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out_dir / "hf_feature_check.png", dpi=130)

summary = dict(N=N, redundancy=redund, eta2_candidates=eta_cand,
               eta2_existing=dict(median=round(float(np.median(eta_exist)), 4),
                                  p90=round(float(np.percentile(eta_exist, 90)), 4),
                                  max=round(float(eta_exist.max()), 4),
                                  top3=sorted([round(float(v), 4) for v in eta_exist])[-3:]))
json.dump(summary, open(out_dir / "hf_feature_check.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("\n== 冗余度（max |r| vs 现有 zfeat） ==")
for n in names:
    print(f"  {n:12s} max|r|={redund[n]['max_abs_r']:.3f}  top3={redund[n]['top']}")
print("\n== η²（36 状态） ==")
for n in names:
    print(f"  {n:12s} η²={eta_cand[n]:.4f}")
print(f"\n现有 zfeat 28 列 η²: 中位={np.median(eta_exist):.4f} "
      f"P90={np.percentile(eta_exist, 90):.4f} 最大={eta_exist.max():.4f}")
print(f"\n图 -> {out_dir / 'hf_feature_check.png'}")
