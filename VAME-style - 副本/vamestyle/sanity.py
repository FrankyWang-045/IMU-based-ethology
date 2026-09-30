# -*- coding: utf-8 -*-
"""v4.0 人工核对物料：联合谱每个 flat 类的代表性信号图。

口径：K=36 sticky-HMM（labels_k36_v12.npy）× 4 姿态桶（方案 2）→ 1% 剪枝
联合谱（joint.py 同源代码，确保与交付谱一致）。每类取置信度最高的
20 个 bout（c = bout 内 sticky 解码后验的母状态后验均值），5×4 排版，
版式对齐 vame_joint_sanity_v2/state_01b.png：
  - 每个 bout 一列：上 acc(g) 三轴（蓝/橙/绿），下 gyro(dps) 三轴
    （红/紫/棕，**不透明**——按用户要求 2026-09-28 修正旧图发虚问题）
  - 黄底带 = bout 实际范围；acc 固定 ±1.5 g，gyro 固定 ±450 °/s
  - 时间轴用 npz 原始时间戳 t（含 120 s 头裁剪的记录原始秒），
    切割位置全部由 t 换算，绝不用采样点次序
  - 子图标题：rec_XXX @<记录原始秒>s c=<置信度>

产物：figures/joint_repr/class_{fid:02d}.png（46 类）
用法：python -m vamestyle.sanity
"""
from pathlib import Path

import joblib
import numpy as np

from vamestyle.dataset import CFG, ROOT, sessions
from vamestyle.joint import K, KB, build_frame_joint, prune
from vamestyle.states import sticky_decode

N_SEG = 20           # 每类代表段数（对齐参考图 20 段/页）
WIN = 3.0            # 窗半宽：bout 中心 ±3 s（共 6 s，对齐参考图坐标 -3..3）
MIN_SEG = 8          # 候选 bout 最短 8 帧 @25 Hz = 320 ms


def _posterior_states(model, z):
    """sticky 模型逐帧状态后验 (T,K)。"""
    return model.predict_proba(z)


def select_segments():
    """每 flat 类选 top-N bout：(fid, session, start, end, conf, t_center)。"""
    frame_joint, bout_rows, usage, _, _ = build_frame_joint()
    remap, _ = prune(usage)
    rec = joblib.load(ROOT / "models" / f"hmm_k{K}_v12.joblib")
    smodel = _sticky_model(rec["model"])

    cands = {}
    C = CFG["vame"]["time_window"] // 2
    ex = set(CFG["data"]["exclude"])
    for s in sessions():
        if s.name in ex:
            continue
        jj = frame_joint[s.name]
        z = np.load(ROOT / "cache" / f"zfeat_{s.name}.npy")
        post = _posterior_states(smodel, z)     # (T,K)
        state = jj // KB
        conf_t = post[np.arange(len(jj)), state]
        f = remap[jj]
        d = np.diff(f)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [len(f)]])
        for a, b in zip(starts, ends):
            if b - a < MIN_SEG:
                continue
            fid = int(f[a])
            conf = float(conf_t[a:b].mean())
            # bout 中心在记录原始时间轴上的秒（t 含 120 s 头裁剪）
            ci = a + (b - a) // 2 + C           # zfeat 行 i ↔ 原始帧 i+15
            tc = float(s.t[ci])
            cands.setdefault(fid, []).append((conf, s.name, int(a), int(b), tc))
    picked = {}
    for fid, lst in cands.items():
        lst.sort(key=lambda x: -x[0])
        picked[fid] = lst[:N_SEG]
    return picked


def _sticky_model(model):
    import copy
    m = copy.deepcopy(model)
    A = m.transmat_ * (1 - 0.98)
    A[np.diag_indices_from(A)] += 0.98
    m.transmat_ = A / A.sum(axis=1, keepdims=True)
    return m


def plot_class(fid, segs, info, outdir):
    """画单个 flat 类的一页（5×4，不足则留空）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    ACC_C = ["#1f77b4", "#ff7f0e", "#2ca02c"]     # 蓝 橙 绿
    GYR_C = ["#d62728", "#9467bd", "#8c564b"]     # 红 紫 棕（不透明）
    n = len(segs)
    nrow, ncol = 4, 5
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(22, 2.1 * nrow * 2))
    st, bk, up = info
    bk_txt = f"桶 {bk}" if bk >= 0 else "桶*（合并）"
    fig.suptitle(f"类 {fid}（state {st} {bk_txt}，{up}%）{n} 段  "
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
            s = [x for x in sessions([name])][0]
            sess_cache[name] = (s.t, s.raw6)
        t, raw6 = sess_cache[name]
        i0, i1 = a + C, b + C                     # bout 在原始帧的 [start, end)
        c = (t[i0] + t[min(i1, len(t) - 1)]) / 2
        w0, w1 = c - WIN, c + WIN
        m = (t >= w0) & (t <= w1)
        tt = t[m] - c
        acc, gyr = raw6[m, 0:3], raw6[m, 3:6]
        for ch in range(3):
            ax_a.plot(tt, acc[:, ch], color=ACC_C[ch], lw=0.7, alpha=1.0)
            ax_g.plot(tt, gyr[:, ch], color=GYR_C[ch], lw=0.7, alpha=1.0)
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
    p = outdir / f"class_{fid:02d}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main():
    outdir = ROOT / "figures" / "joint_repr"
    outdir.mkdir(parents=True, exist_ok=True)
    picked = select_segments()
    # flat 类元信息（state/bucket/usage）
    import csv as _csv
    spec = {int(r["flat_id"]): (int(r["state"]), int(r["bucket"]),
                                float(r["usage_pct"]))
            for r in _csv.DictReader(open(
                ROOT / "results" / f"joint_spectrum_k{K}_v12_v2.csv",
                encoding="utf-8"))}
    for fid in sorted(picked):
        p = plot_class(fid, picked[fid], spec[fid], outdir)
        print(f"[sanity] 类 {fid}: {len(picked[fid])} 段 -> {p.name}")


if __name__ == "__main__":
    main()
