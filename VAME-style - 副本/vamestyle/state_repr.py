# -*- coding: utf-8 -*-
"""v4.0 人工核对物料（对照组）：36 状态单层分割的代表性信号图。

与 sanity.py（46 flat 类 = 状态×4 姿态桶 + 1% 剪枝）并存，供用户对比
"组合+剪枝"与"单层直接分割"两种口径。本脚本**不经姿态桶、不经剪枝**：
标签 = sticky 解码状态的原始 36 值（frame_joint // KB）。
每状态取置信度最高的 20 个 bout，5×4 排版，版式与 sanity.py 完全一致：
  - 每 bout 一列：上 acc(g) 三轴（蓝/橙/绿），下 gyro(dps) 三轴
    （红/紫/棕，不透明）
  - 黄底带 = bout；acc 固定 ±1.5 g，gyro 固定 ±450 °/s
  - 时间轴用 npz 原始时间戳 t，切割位置全部由 t 换算
产物：figures/state_repr/state_{sid:02d}.png（36 张）
      figures/state_repr_c/state_{sid:02d}.png（路线 C 对照：labels_k36_v12_c）
用法：python -m vamestyle.state_repr      # 旧 28 维 zfeat 口径
      python -m vamestyle.state_repr c    # 路线 C 29 维 zfeat 口径
"""
from pathlib import Path

import joblib
import numpy as np

from vamestyle.dataset import CFG, ROOT, sessions
from vamestyle.joint import K, KB, build_frame_joint
from vamestyle.sanity import _sticky_model

N_SEG = 20
WIN = 3.0
MIN_SEG = 8


def select_segments(version=None):
    """每状态 top-N bout：(sid, session, start, end, conf, t_center)。
    version="c"：路线 C 口径，直接读 labels_k36_v12_c.npy 与 zfeat_c。"""
    C = CFG["vame"]["time_window"] // 2
    ex = set(CFG["data"]["exclude"])
    cands = {}
    usage = {}
    if version == "c":
        rec = joblib.load(ROOT / "models" / f"hmm_k{K}_v12_c.joblib")
        smodel = _sticky_model(rec["model"])
        labels = {}
        for s in sessions():
            if s.name in ex:
                continue
            labels[s.name] = np.load(
                ROOT / "results" / s.name / f"labels_k{K}_v12_c.npy")
        zcache = {s.name: np.load(ROOT / "cache" / f"zfeat_c_{s.name}.npy")
                  for s in sessions() if s.name not in ex}
    else:
        frame_joint, _, _, _, _ = build_frame_joint()
        rec = joblib.load(ROOT / "models" / f"hmm_k{K}_v12.joblib")
        smodel = _sticky_model(rec["model"])
        labels = {s.name: frame_joint[s.name] // KB
                  for s in sessions() if s.name not in ex}
        zcache = {s.name: np.load(ROOT / "cache" / f"zfeat_{s.name}.npy")
                  for s in sessions() if s.name not in ex}
    for s in sessions():
        if s.name in ex:
            continue
        state = labels[s.name]
        z = zcache[s.name]
        post = smodel.predict_proba(z)        # (T,K)
        conf_t = post[np.arange(len(state)), state]
        for sid in np.unique(state):
            usage[int(sid)] = usage.get(int(sid), 0) + int((state == sid).sum())
        d = np.diff(state)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [len(state)]])
        for a, b in zip(starts, ends):
            if b - a < MIN_SEG:
                continue
            sid = int(state[a])
            conf = float(conf_t[a:b].mean())
            ci = a + (b - a) // 2 + C
            tc = float(s.t[ci])
            cands.setdefault(sid, []).append((conf, s.name, int(a), int(b), tc))
    picked = {}
    for sid, lst in cands.items():
        lst.sort(key=lambda x: -x[0])
        picked[sid] = lst[:N_SEG]
    tot = sum(usage.values())
    usage_pct = {sid: 100.0 * n / tot for sid, n in usage.items()}
    return picked, usage_pct


def plot_state(sid, segs, up, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ACC_C = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    GYR_C = ["#d62728", "#9467bd", "#8c564b"]
    n = len(segs)
    nrow, ncol = 4, 5
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(22, 2.1 * nrow * 2))
    fig.suptitle(f"状态 {sid}（36 状态单层分割，{up:.1f}%）{n} 段  "
                 f"上=acc(g)±1.5 下=gyro(dps)±450", fontsize=13)
    C = CFG["vame"]["time_window"] // 2
    sess_cache = {}
    for j in range(nrow * ncol):
        ax_a = axes[2 * (j // ncol), j % ncol]
        ax_g = axes[2 * (j // ncol) + 1, j % ncol]
        if j >= n:
            ax_a.axis("off"); ax_g.axis("off")
            continue
        conf, name, a, b, tc = segs[j]
        if name not in sess_cache:
            s = sessions([name])[0]
            sess_cache[name] = (s.t, s.raw6)
        t, raw6 = sess_cache[name]
        i0, i1 = a + C, b + C
        c = (t[i0] + t[min(i1, len(t) - 1)]) / 2
        m = (t >= c - WIN) & (t <= c + WIN)
        tt = t[m] - c
        for ch in range(3):
            ax_a.plot(tt, raw6[m, ch], color=ACC_C[ch], lw=0.7, alpha=1.0)
            ax_g.plot(tt, raw6[m, ch + 3], color=GYR_C[ch], lw=0.7, alpha=1.0)
        for ax in (ax_a, ax_g):
            ax.axvspan(t[i0] - c, t[min(i1, len(t) - 1)] - c,
                       color="orange", alpha=0.25, lw=0)
            ax.set_xlim(-WIN, WIN)
            ax.tick_params(labelsize=6)
        ax_a.set_ylim(-1.5, 1.5)
        ax_g.set_ylim(-450, 450)
        ax_a.set_title(f"{name} @{tc:.1f}s c={conf:.2f}", fontsize=8)
        if j % ncol == 0:
            ax_a.set_ylabel("acc(g)", fontsize=7)
            ax_g.set_ylabel("gyro(dps)", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = outdir / f"state_{sid:02d}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main(version=None):
    outdir = ROOT / "figures" / ("state_repr_c" if version == "c" else "state_repr")
    outdir.mkdir(parents=True, exist_ok=True)
    picked, usage_pct = select_segments(version)
    for sid in sorted(picked):
        p = plot_state(sid, picked[sid], usage_pct.get(sid, 0.0), outdir)
        print(f"[state_repr] 状态 {sid}: {len(picked[sid])} 段 "
              f"usage={usage_pct.get(sid, 0):.1f}% -> {p.name}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
