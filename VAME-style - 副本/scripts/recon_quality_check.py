# -*- coding: utf-8 -*-
"""重构质量定量检查：逐通道 R²、分频段 R²、输入/重构功率谱对比。
用法：v3_recovery venv python scripts/recon_quality_check.py [run_dir]
"""
import sys, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(r"F:\Kimi\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
VAME_IMU = WORKSPACE / "VAME-IMU"
for p in (str(VAME_STYLE), str(VAME_IMU)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from vame.model.rnn_vae import RNN_VAE

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else VAME_STYLE / "runs" / "manual_20260928_140939"
PROJ = RUN_DIR / "vame_project"

# 与 §1 面板默认（dyn6）一致
TW, ZDIMS, NF = 30, 16, 6
model = RNN_VAE(TW * 2, ZDIMS, NF, True, 15,
                256, 256, 64, 64, 0.2, 0.2, 0.2, True).to(DEVICE)
best_p = PROJ / "model" / "best_model" / "VAME_vame_project.pkl"
model.load_state_dict(torch.load(best_p, map_location=DEVICE))
model.eval()

td = PROJ / "data" / "train"
Xte = np.load(td / "test_seq.npy")
if Xte.shape[0] > Xte.shape[1]:
    Xte = Xte.T
mean = float(np.load(td / "seq_mean.npy"))
std = float(np.load(td / "seq_std.npy"))
Xte = ((Xte - mean) / std).astype(np.float32)
print(f"test_seq {Xte.shape}, mean={mean:.4f}, std={std:.4f}, device={DEVICE}")

# 滑窗取样本（stride=7，约 4 s 间隔），batch 前向
rng = np.random.default_rng(0)
starts = np.arange(0, Xte.shape[1] - TW, 7)
N = len(starts)
CH_NAMES = ["linaccx", "linaccy", "linaccz", "gyrox", "gyroy", "gyroz"]
rec = np.empty((N, TW, NF), dtype=np.float32)
raw = np.empty((N, TW, NF), dtype=np.float32)
B = 512
with torch.no_grad():
    for i in range(0, N, B):
        idx = starts[i:i + B]
        w = np.stack([Xte[:, s:s + TW].T for s in idx])
        wt = torch.from_numpy(w).float().to(DEVICE)
        out = model(wt)
        rec[i:i + len(idx)] = out[0][:, :, :].cpu().numpy()
        raw[i:i + len(idx)] = w
print(f"重构窗口数 N={N}")

# ---- 逐通道 Pearson r 与 R² ----
per_ch = {}
for c in range(NF):
    x, r = raw[:, :, c].ravel(), rec[:, :, c].ravel()
    r_pearson = float(np.corrcoef(x, r)[0, 1])
    r2 = 1.0 - float(((x - r) ** 2).sum() / ((x - x.mean()) ** 2).sum())
    per_ch[CH_NAMES[c]] = dict(pearson=round(r_pearson, 4), r2=round(r2, 4),
                               var=round(float(x.var()), 3))

# ---- 分频段 R²（rFFT 带通后比较） ----
freqs = np.fft.rfftfreq(TW, d=1 / 25.0)
bands = [("0–1.7 Hz", freqs < 1.7), ("1.7–5 Hz", (freqs >= 1.7) & (freqs < 5.0)),
         ("5–8.3 Hz", (freqs >= 5.0) & (freqs < 8.3)), ("8.3–12.5 Hz", freqs >= 8.3)]
Xf = np.fft.rfft(raw, axis=1)
Rf = np.fft.rfft(rec, axis=1)
band_r2 = {}
for bname, mask in bands:
    cols = {}
    for c in range(NF):
        xb = np.fft.irfft(Xf[:, mask, c], n=TW, axis=1)
        rb = np.fft.irfft(Rf[:, mask, c], n=TW, axis=1)
        sse = float(((xb - rb) ** 2).sum())
        sst = float(((xb - xb.mean()) ** 2).sum())
        cols[CH_NAMES[c]] = round(1.0 - sse / sst, 4)
    band_r2[bname] = cols

# ---- 平均功率谱（重构/输入 衰减比） ----
Px = (np.abs(Xf) ** 2).mean(axis=0)   # (F, 6)
Pr = (np.abs(Rf) ** 2).mean(axis=0)
ratio_db = 10 * np.log10((Pr + 1e-12) / (Px + 1e-12))

# ---- 图 ----
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.1, 1.1], hspace=0.45)
ax1 = fig.add_subplot(gs[0])
xs = np.arange(NF)
r2s = [per_ch[n]["r2"] for n in CH_NAMES]
prs = [per_ch[n]["pearson"] for n in CH_NAMES]
ax1.bar(xs - 0.18, r2s, width=0.36, label="R²", color="#4C7FB0")
ax1.bar(xs + 0.18, prs, width=0.36, label="Pearson r", color="#D9A441")
for i, (a, b) in enumerate(zip(r2s, prs)):
    ax1.text(i - 0.18, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    ax1.text(i + 0.18, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
ax1.set_xticks(xs, CH_NAMES)
ax1.set_ylim(0, 1.15)
ax1.set_title(f"逐通道重建质量（N={N} 窗，test_seq，标准化域）")
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[1])
width = 0.19
for bi, (bname, _) in enumerate(bands):
    vals = [band_r2[bname][n] for n in CH_NAMES]
    ax2.bar(xs + (bi - 1.5) * width, vals, width=width, label=bname)
ax2.set_xticks(xs, CH_NAMES)
ax2.axhline(0, color="k", lw=0.5)
ax2.set_title("分频段 R²（负值 = 重构在该频段比均值预测还差）")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[2])
for c in range(NF):
    ax3.plot(freqs, ratio_db[:, c], lw=1.0, label=CH_NAMES[c])
ax3.axhline(0, color="k", lw=0.8, ls="--")
ax3.set_xlabel("频率 (Hz)")
ax3.set_ylabel("重构/输入 功率比 (dB)")
ax3.set_title("平均功率谱衰减（<0 dB = 重构在该频率能量低于输入）")
ax3.set_ylim(-30, 5)
ax3.legend(ncol=3, fontsize=8); ax3.grid(alpha=0.3)

out_png = RUN_DIR / "figures" / "recon_quality_check.png"
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=130, bbox_inches="tight")

summary = dict(run=str(RUN_DIR), n_windows=N, per_channel=per_ch,
               band_r2=band_r2,
               power_ratio_db={CH_NAMES[c]: [round(float(v), 1) for v in ratio_db[::4, c]]
                               for c in range(NF)},
               freq_grid=[round(float(f), 2) for f in freqs[::4]])
out_json = RUN_DIR / "outputs" / "recon_quality_check.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("\n逐通道:", json.dumps(per_ch, ensure_ascii=False))
print("\n分频段 R²:")
for b, cols in band_r2.items():
    print(f"  {b}: " + " ".join(f"{n}={v:.2f}" for n, v in cols.items()))
print(f"\n图 -> {out_png}\n摘要 -> {out_json}")
