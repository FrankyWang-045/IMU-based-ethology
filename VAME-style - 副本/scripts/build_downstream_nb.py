# -*- coding: utf-8 -*-
"""生成 downstream_validation.ipynb（下游验证 notebook #2）。

用法：python scripts/build_downstream_nb.py
"""
import json
from pathlib import Path

NB_PATH = Path(r"F:\Kimi\IMU\VAME-Style\downstream_validation.ipynb")

CELLS = [
    ("markdown", """# VAME-Style v4.0 下游验证（notebook #2）

**内核**：`v3_recovery` venv 同 manual_train（仅需 numpy/scipy/sklearn/matplotlib/pandas，无需 GPU）。
**数据源（全部只读，均为生产定档）**：`cache/mu_*`（vame_25hz_14s 最佳 ckpt）、生产 `results/<sess>/labels_k36_v12.npy`（K=36 单层 sticky）、ds25 npz。
**四部分**：① 段级手工特征（不含姿态）→ ② 类内相似度置换检验（零分布直方图 + 逐类 t 值）→ ③ 类×特征热图（Cohen's d / z）→ ④ 类间 latent 相似度矩阵。
**规范**：只在 §1 改参数；每次运行全部产物落在 `run_dir` 单一目录（figures/ outputs/ logs/）；绘图 cell 与计算解耦，可反复重跑；§10 生成日志。"""),

    ("code", """# ==== §0 环境与全局路径 ====
import sys, json, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

WORKSPACE = Path(r"D:\\KimiData\\kimi\\Workspaces\\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
VAME_IMU = WORKSPACE / "VAME-IMU"
for p in (str(VAME_STYLE), str(VAME_IMU)):
    if p not in sys.path:
        sys.path.insert(0, p)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    matplotlib.use("Agg")

from vamestyle.dataset import CFG, ROOT, sessions   # 复用工程配置（exclude/fs 等）

FS = float(CFG["data"]["fs"])
EXCLUDE = set(CFG["data"].get("exclude", []))       # v12 口径：rec_011/013
print("fs =", FS, "| exclude =", sorted(EXCLUDE))"""),

    ("code", """# ==== §1 参数面板（唯一改参处） ====
P = dict(
    # ---- 输出根目录：每次运行手填（默认按时间戳新建）----
    run_dir=str(VAME_STYLE / "runs" / f"val_{datetime.now():%Y%m%d_%H%M%S}"),

    data=dict(
        npz_dir=str(VAME_IMU / "data" / "ds25"),     # 25 Hz npz（raw6 / t）
        labels_tmpl="results/{sess}/labels_k36_v12.npy",  # 生产单层 36 状态标签
        mu_dir=str(ROOT / "cache"),                  # 生产 mu（vame_25hz_14s）
        subset="v12",                                # 排除 data.exclude
    ),

    seg=dict(min_sec=0.5),      # 段最短时长（秒）；方差/过零率统计稳定性下限

    feat=dict(
        detrend=True,           # 段内逐轴线性去趋势后再算统计量
        bands=[(0.0, 2.0), (2.0, 5.0), (5.0, 8.333), (8.333, 12.5)],
    ),

    test=dict(
        n_sample=1000,          # 分层抽样总段数
        k_nn=10,                # k-NN 纯度近邻数
        n_perm=1000,            # 置换次数（标签重排，类大小边际不变）
        seed=0,
    ),

    plot=dict(dpi=130),
)


def apply_params():
    \"\"\"由 P 派生路径常量并创建目录层级（改 P 后重跑本函数即可）。\"\"\"
    global RUN_DIR, FIG_DIR, OUT_DIR, LOG_DIR, MIN_LEN
    RUN_DIR = Path(P["run_dir"])
    FIG_DIR = RUN_DIR / "figures"
    OUT_DIR = RUN_DIR / "outputs"
    LOG_DIR = RUN_DIR / "logs"
    for d in (FIG_DIR, OUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    MIN_LEN = max(4, int(round(P["seg"]["min_sec"] * FS)))
    print("run_dir =", RUN_DIR)
    print("段最短长度 =", MIN_LEN, f"帧（{P['seg']['min_sec']} s）")


apply_params()"""),

    ("code", """# ==== §2 数据装载（生产缓存，只读） ====
NPZ_DIR = Path(P["data"]["npz_dir"])
MU_DIR = Path(P["data"]["mu_dir"])

data = {}
for s in sessions():
    if s.name in EXCLUDE:
        continue
    d = np.load(NPZ_DIR / f"{s.name}.npz")
    lab = np.load(ROOT / P["data"]["labels_tmpl"].format(sess=s.name))
    mu = np.load(MU_DIR / f"mu_{s.name}.npy")
    assert len(lab) == len(mu), (s.name, len(lab), len(mu))
    assert len(d["raw6"]) == len(lab) + 29, s.name   # zfeat 行 i ↔ 原始帧 i+15
    data[s.name] = dict(t=d["t"].astype(np.float64),
                        dyn6=d["raw6"].astype(np.float64),
                        labels=lab.astype(np.int16), mu=mu.astype(np.float64))
    print(f"[data] {s.name}: T={len(d['raw6'])} 段标签={len(lab)} mu={mu.shape}")

K = int(max(np.concatenate([dd["labels"] for dd in data.values()])) + 1)
print(f"session 数 = {len(data)}，状态数 K = {K}")"""),

    ("code", """# ==== §3 bout 段提取（sticky 标签游程切分 + 最短时长过滤） ====
rows = []
for name, dd in data.items():
    lab = dd["labels"]
    d = np.diff(lab)
    starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
    ends = np.concatenate([starts[1:], [len(lab)]])
    for a, b in zip(starts, ends):
        if b - a < MIN_LEN:
            continue
        rows.append((name, int(a), int(b), int(lab[a]), int(b - a)))
seg_df = pd.DataFrame(rows, columns=["session", "start", "end", "state", "dur"])
seg_df["sec"] = seg_df["dur"] / FS
seg_df.to_csv(OUT_DIR / "segments.csv", index=False)
print(f"段总数 = {len(seg_df)}（已过滤 < {MIN_LEN} 帧）")
print("每类段数分布：min={} / 中位={:.0f} / max={}".format(
    seg_df.groupby('state').size().min(),
    seg_df.groupby('state').size().median(),
    seg_df.groupby('state').size().max()))"""),

    ("code", """# ==== §4 段级手工特征（不含姿态；acc/gyro 两组；输出 robust-z） ====
BANDS = [tuple(b) for b in P["feat"]["bands"]]
GRP = [("acc", slice(0, 3)), ("gyr", slice(3, 6))]
EPS = 1e-12


def _detrend(x):
    \"\"\"(L,C) 逐轴线性去趋势。\"\"\"
    L = len(x)
    t = np.arange(L)
    out = x.copy()
    for c in range(x.shape[1]):
        k, m = np.polyfit(t, x[:, c], 1)
        out[:, c] = x[:, c] - (k * t + m)
    return out


def seg_features(dyn6, a, b):
    \"\"\"dyn6 帧区间 [a+15, b+15)（zfeat 对齐）→ 特征向量（未标准化）。\"\"\"
    C = 15
    x_raw = dyn6[a + C:b + C]                # (L,6)
    x = _detrend(x_raw) if P["feat"]["detrend"] else x_raw
    L = len(x)
    f = dict(length_frames=float(L), length_sec=L / FS)
    freqs = np.fft.rfftfreq(L, d=1 / FS)
    for gname, sl in GRP:
        xg = x[:, sl]                        # (L,3) 已去趋势
        f[f"{gname}_log_energy"] = np.log10(np.mean(np.sum(xg * xg, axis=1)) + EPS)
        Pw = (np.abs(np.fft.rfft(xg, axis=0)) ** 2).sum(axis=1)   # 轴求和功率谱
        tot = Pw.sum() + EPS
        for i, (lo, hi) in enumerate(BANDS):
            m = (freqs >= lo) & (freqs < hi if i < len(BANDS) - 1 else freqs <= hi)
            f[f"{gname}_band{i}_{lo:g}-{hi:g}Hz"] = Pw[m].sum() / tot
        for i, ax in enumerate("xyz"):
            f[f"{gname}_mean_{ax}"] = x_raw[:, sl][:, i].mean()      # 均值用原始（保偏移）
            f[f"{gname}_log_var_{ax}"] = np.log10(xg[:, i].var() + EPS)
        sgn = np.sign(xg)
        zcr = (np.diff(sgn, axis=0) != 0).mean(axis=1).mean()
        f[f"{gname}_zcr"] = float(zcr)
    return f


FEAT_ROWS = []
for _, r in seg_df.iterrows():
    dd = data[r["session"]]
    FEAT_ROWS.append(seg_features(dd["dyn6"], r["start"], r["end"]))
feat_df = pd.DataFrame(FEAT_ROWS)
FEAT_NAMES = [c for c in feat_df.columns if c != "length_frames"]
# length_frames 参与距离但不进热图（量纲差太多，robust-z 后等同 length_sec，删一留一）
FEAT_NAMES.remove("length_sec")


def _robust_z_df(df, cols):
    med = df[cols].median()
    q75, q25 = df[cols].quantile(0.75), df[cols].quantile(0.25)
    iqr = (q75 - q25).replace(0, 1.0)
    return ((df[cols] - med) / iqr).astype(np.float32)


Fz = _robust_z_df(feat_df, FEAT_NAMES)
Fz.insert(0, "length_sec", feat_df["length_sec"])
Fz.insert(0, "state", seg_df["state"].values)
Fz.insert(0, "session", seg_df["session"].values)
Fz.to_csv(OUT_DIR / "segment_features.csv", index=False)
X = Fz[FEAT_NAMES].to_numpy(np.float32)          # (S,D) robust-z
y = seg_df["state"].to_numpy(np.int16)
print(f"特征矩阵 X={X.shape}，共 {len(FEAT_NAMES)} 维：")
print(FEAT_NAMES)"""),

    ("code", """# ==== §5 分层抽样（按类封顶 + 轮转补足，seed 固定） ====
def stratified_sample(y, n_total, seed):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    cap = int(np.ceil(n_total / len(classes)))
    take = {c: min(n, cap) for c, n in zip(classes, counts)}
    deficit = n_total - sum(take.values())
    if deficit > 0:                          # 未满额的大类轮转补足
        room = sorted(((counts[list(classes).index(c)] - take[c], c)
                       for c in classes), key=lambda x: -x[0])
        while deficit > 0 and any(r > 0 for r, _ in room):
            for r, c in room:
                if deficit <= 0:
                    break
                if r > 0:
                    take[c] += 1
                    deficit -= 1
            room = [(counts[list(classes).index(c)] - take[c], c) for _, c in room]
    idx = np.concatenate([rng.choice(np.where(y == c)[0], take[c], replace=False)
                          for c in classes])
    if len(idx) > n_total:                      # cap 超额 → 等比例缩回
        idx = rng.choice(idx, n_total, replace=False)
    rng.shuffle(idx)
    return idx


SAMPLE_IDX = stratified_sample(y, P["test"]["n_sample"], P["test"]["seed"])
assert len(SAMPLE_IDX) == P["test"]["n_sample"]
np.save(OUT_DIR / "sample_idx.npy", SAMPLE_IDX)
print("抽样 =", len(SAMPLE_IDX),
      "| 覆盖类数 =", len(np.unique(y[SAMPLE_IDX])),
      "| 每类上限 ≈", int(np.ceil(P['test']['n_sample'] / K)))"""),

    ("code", """# ==== §6 置换检验（标签重排 = 随机分配零假设；BH-FDR） ====
from scipy.spatial.distance import pdist, squareform

D = squareform(pdist(X[SAMPLE_IDX])).astype(np.float64)   # (N,N) 欧氏
Ls = y[SAMPLE_IDX]
N = len(Ls)
KNN = P["test"]["k_nn"]
N_PERM = P["test"]["n_perm"]


def mean_silhouette(D, lab):
    \"\"\"全体平均轮廓系数：s=(b-a)/max(a,b)。\"\"\"
    same = lab[:, None] == lab[None, :]
    cnt = np.bincount(lab)
    a = (D * same).sum(axis=1) / np.maximum(cnt[lab] - 1, 1)
    b = np.full(N, np.inf)
    for c in np.unique(lab):
        m = lab == c
        if m.sum() == 0:
            continue
        mean_d = (D * m[None, :]).sum(axis=1) / m.sum()
        mean_d[m] = np.inf                        # 同类不算 b
        b = np.minimum(b, mean_d)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    return float(s.mean())


def knn_purity(D, lab):
    nn = np.argsort(D, axis=1)[:, 1:KNN + 1]      # 不含自身
    return float((lab[nn] == lab[:, None]).mean())


def class_silhouettes(D, lab):
    \"\"\"逐类平均 silhouette（类内点的 s 均值）。\"\"\"
    same = lab[:, None] == lab[None, :]
    cnt = np.bincount(lab)
    a = (D * same).sum(axis=1) / np.maximum(cnt[lab] - 1, 1)
    b = np.full(N, np.inf)
    for c in np.unique(lab):
        m = lab == c
        mean_d = (D * m[None, :]).sum(axis=1) / m.sum()
        mean_d[m] = np.inf
        b = np.minimum(b, mean_d)
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
    out = {}
    for c in np.unique(lab):
        out[int(c)] = float(s[lab == c].mean())
    return out


rng = np.random.default_rng(P["test"]["seed"])
obs_sil = mean_silhouette(D, Ls)
obs_pur = knn_purity(D, Ls)
obs_cls = class_silhouettes(D, Ls)
null_sil = np.empty(N_PERM)
null_pur = np.empty(N_PERM)
null_cls = {c: np.empty(N_PERM) for c in obs_cls}
t0 = time.time()
for i in range(N_PERM):
    lp = rng.permutation(Ls)
    null_sil[i] = mean_silhouette(D, lp)
    null_pur[i] = knn_purity(D, lp)
    cs = class_silhouettes(D, lp)
    for c in null_cls:
        null_cls[c][i] = cs.get(c, np.nan)
    if (i + 1) % 200 == 0:
        print(f"[perm] {i+1}/{N_PERM}  [{time.time()-t0:.0f}s]", flush=True)

emp_p_sil = float((null_sil >= obs_sil).mean())
emp_p_pur = float((null_pur >= obs_pur).mean())
# 逐类 t 值 =（观测 − 零分布均值）/ 零分布 std；单侧经验 p + BH-FDR
cls_rows = []
pvals = []
for c, obs in obs_cls.items():
    nz = null_cls[c][np.isfinite(null_cls[c])]
    t_v = (obs - nz.mean()) / (nz.std() + 1e-12)
    p_v = float((nz >= obs).mean())
    pvals.append(p_v)
    cls_rows.append((c, obs, t_v, p_v))
# Benjamini-Hochberg
order = np.argsort([r[3] for r in cls_rows])
m = len(cls_rows)
q = np.empty(m)
prev = 1.0
for rank, oi in enumerate(order[::-1], start=1):
    i = m - rank                              # 0-based 从大到小
    prev = min(prev, pvals[oi] * m / (i + 1))
    q[oi] = prev
cls_stat = pd.DataFrame(
    [(r[0], r[1], r[2], r[3], float(qq)) for r, qq in zip(cls_rows, q)],
    columns=["state", "silhouette", "t_value", "p_emp", "q_fdr"])
cls_stat.to_csv(OUT_DIR / "class_permutation_stats.csv", index=False)
np.savez_compressed(OUT_DIR / "perm.npz", D=D, labels=Ls,
                    null_sil=null_sil, null_pur=null_pur,
                    obs_sil=obs_sil, obs_pur=obs_pur,
                    **{f"null_cls_{c}": v for c, v in null_cls.items()})
print(f"全局 silhouette = {obs_sil:.4f}（置换 p = {emp_p_sil:.4f}）")
print(f"k-NN 纯度(k={KNN}) = {obs_pur:.3f}（置换 p = {emp_p_pur:.4f}）")
print(f"逐类：显著(FDR<0.05) {int((cls_stat.q_fdr<0.05).sum())}/{len(cls_stat)}")"""),

    ("code", """# ==== §7 图 2a（零分布直方图）+ 图 2b（逐类 t 值条形图） ====
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
ax = axes[0]
ax.hist(null_sil, bins=40, color="#9AA0A6", alpha=0.75, label="随机分配零分布")
ax.axvline(obs_sil, color="#B45309", lw=2, label=f"观测值 {obs_sil:.3f}")
ax.set_title(f"全局平均 silhouette（置换 p={emp_p_sil:.4f}）")
ax.set_xlabel("mean silhouette"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.hist(null_pur, bins=40, color="#9AA0A6", alpha=0.75, label="随机分配零分布")
ax.axvline(obs_pur, color="#B45309", lw=2, label=f"观测值 {obs_pur:.3f}")
ax.set_title(f"k-NN 纯度 k={KNN}（置换 p={emp_p_pur:.4f}）")
ax.set_xlabel("purity"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "perm_null_hist.png", dpi=P["plot"]["dpi"])
plt.show()

fig, ax = plt.subplots(figsize=(14, 4.6))
sig = cls_stat.q_fdr.values < 0.05
colors = np.where(sig, "#B45309", "#9CA3AF")
ax.bar(cls_stat.state, cls_stat.t_value, color=colors)
ax.axhline(0, color="k", lw=0.6)
ax.axhline(1.96, color="#B45309", ls="--", lw=0.8)
ax.axhline(-1.96, color="#B45309", ls="--", lw=0.8)
ax.set_xticks(cls_stat.state)
ax.set_xlabel("状态"); ax.set_ylabel("t 值（相对随机分配）")
ax.set_title(f"逐类类内紧致性 t 值（琥珀=FDR<0.05 显著，灰=不显著；"
             f"显著 {sig.sum()}/{len(cls_stat)}）")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG_DIR / "perm_class_tvals.png", dpi=P["plot"]["dpi"])
plt.show()"""),

    ("code", """# ==== §8 类×手工特征热图（Cohen's d 主 + 跨类 z 辅；全量段，类中位数） ====
from scipy.cluster import hierarchy

med = feat_df.assign(state=y).groupby("state")[FEAT_NAMES].median()
glob_med = feat_df[FEAT_NAMES].median()
glob_std = feat_df[FEAT_NAMES].std().replace(0, 1.0)
Z = ((med - glob_med) / glob_std)                       # 跨类 z（类中位数口径）

Drows = []
for c in med.index:
    a = feat_df.loc[y == c, FEAT_NAMES]
    b = feat_df.loc[y != c, FEAT_NAMES]
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var())
                 / (len(a) + len(b) - 2)).replace(0, 1e-12)
    Drows.append((a.median() - b.median()) / sp)
Dmat = pd.DataFrame(Drows, index=med.index, columns=FEAT_NAMES)

link = hierarchy.linkage(Z.to_numpy(), method="average")
row_order = hierarchy.leaves_list(link)


def _heat(mat, title, fname, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(13, 8))
    im = ax.imshow(mat.to_numpy()[row_order], aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)), mat.columns, rotation=45,
                  ha="right", fontsize=7)
    ax.set_yticks(range(len(mat.index)), mat.index[row_order], fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=P["plot"]["dpi"])
    plt.show()


_heat(Dmat, "类×手工特征 Cohen's d（该类中位数 vs 其余，行=层次聚类）",
      "heatmap_hand_d.png", "RdBu_r", -1.5, 1.5)
_heat(Z, "类×手工特征 跨类 z-score（类中位数）", "heatmap_hand_z.png",
      "RdBu_r", -2.0, 2.0)
Z.to_csv(OUT_DIR / "heatmap_z.csv"); Dmat.to_csv(OUT_DIR / "heatmap_d.csv")"""),

    ("code", """# ==== §9 类间 latent 相似度矩阵（段级 mu 均值 → 类中位数质心 → 余弦） ====
MU_SEG = np.empty((len(seg_df), data[next(iter(data))]["mu"].shape[1]),
                  dtype=np.float64)
for i, r in seg_df.iterrows():
    MU_SEG[i] = data[r["session"]]["mu"][r["start"]:r["end"]].mean(axis=0)
CEN = pd.DataFrame(MU_SEG).assign(state=y).groupby("state").median()
Cn = (CEN - CEN.mean()) / CEN.std().replace(0, 1.0)     # 质心逐维标准化
S = (Cn @ Cn.T) / (np.linalg.norm(Cn, axis=1)[:, None]
                   * np.linalg.norm(Cn, axis=1)[None, :])
np.save(OUT_DIR / "latent_sim_matrix.npy", S.to_numpy())

link2 = hierarchy.linkage(S.to_numpy(), method="average")
o2 = hierarchy.leaves_list(link2)
fig, ax = plt.subplots(figsize=(9.5, 8.5))
im = ax.imshow(S.to_numpy()[np.ix_(o2, o2)], cmap="RdBu_r",
               vmin=-1, vmax=1)
ax.set_xticks(range(len(S)), S.index[o2], fontsize=7)
ax.set_yticks(range(len(S)), S.index[o2], fontsize=7)
ax.set_title("类间相似度矩阵（mu 类质心余弦，行/列=层次聚类）")
fig.colorbar(im, ax=ax, shrink=0.7)
fig.tight_layout()
fig.savefig(FIG_DIR / "latent_sim_matrix.png", dpi=P["plot"]["dpi"])
plt.show()"""),

    ("code", """# ==== §10 日志汇总（每次实验结束运行一次） ====
summary = dict(
    run_dir=str(RUN_DIR), params=P, K=K,
    n_segments=int(len(seg_df)),
    global_silhouette=obs_sil, p_silhouette=emp_p_sil,
    knn_purity=obs_pur, p_purity=emp_p_pur,
    n_classes_sig_fdr=int((cls_stat.q_fdr < 0.05).sum()),
    class_stats_csv=str(OUT_DIR / "class_permutation_stats.csv"),
)
json.dump(summary, open(LOG_DIR / "validation_summary.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
with open(LOG_DIR / "validation_done.txt", "w", encoding="utf-8") as f:
    f.write(f"下游验证完成\\nrun_dir={RUN_DIR}\\n"
            f"段数={len(seg_df)}  全局silhouette={obs_sil:.4f}(p={emp_p_sil:.4f})"
            f"  kNN纯度={obs_pur:.3f}(p={emp_p_pur:.4f})\\n"
            f"逐类显著(FDR<0.05)={int((cls_stat.q_fdr<0.05).sum())}/{len(cls_stat)}\\n"
            f"耗时统计见各 cell 输出\\n")
print("VALIDATION_DONE")
print(json.dumps({k: v for k, v in summary.items() if k != "params"},
                 ensure_ascii=False, indent=2))"""),
]


def main():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "v3rec",
                           "language": "python", "name": "v3rec"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for ctype, src in CELLS:
        cell = {"cell_type": ctype,
                "metadata": {},
                "source": src.splitlines(keepends=True)}
        if ctype == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb["cells"].append(cell)
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print("written ->", NB_PATH, f"({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
