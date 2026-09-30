# -*- coding: utf-8 -*-
"""v4.0 姿态流形：bout 级 grav_mean 的相对姿态分布与分桶（Phase 6 第①②步）。

参考系（用户拍板 2026-09-28，方案 2 = per_mouse_v2）：
  每 session 以其自身重力中位对齐垂直方向（grav 沿 -z，v0=(0,0,-1)），
  最小弧旋转 R_s 施加到该 session 全部 grav 帧后再算 bout 均值——
  跨鼠可比，且不丢原始信息（gx,gy,gz 列全程保留未旋转原值）。
  ref="v0_global" 保留旧全局口径用于对照。

口径：
  bout：labels_k<K>[_v12] 游程切分，最短 4 帧；zfeat 行 i ↔ 原始帧 i+15。
  grav_mean：bout 内 grav3 均值（原尺度）；grav_std：帧间漂移 L2 均值。
  相对姿态 δ = rotated(grav_mean) − v0；可视化 PCA→PC1/PC2；
  θ = 偏离自己垂直位的倾角，φ = 倾斜方位。

产物：
  cache/posture_bouts_k<K>[_v2].csv   bout 总表（raw + rotated grav 列）
  figures/posture_manifold_k<K>[_v2].png
  figures/posture_k_compare[_v2].png
  results/posture_k_metrics[_v2].csv

用法：python -m vamestyle.posture 36 v2      （K, ref；默认 36 v2）
"""
import csv
import sys
from pathlib import Path

import numpy as np

from vamestyle.dataset import CFG, ROOT, sessions
from vamestyle.states import MIN_BOUT

K_DEFAULT = 36
V0 = np.array([0.0, 0.0, -1.0])          # 重力垂直位（grav 沿 -z）
K_CANDIDATES = [3, 4, 6]


def _align_rotation(q):
    """单位向量 q → v0 的最小弧旋转矩阵（Rodrigues）。"""
    q = q / np.linalg.norm(q)
    axis = np.cross(q, V0)
    na = np.linalg.norm(axis)
    if na < 1e-9:                        # 已对齐或反向
        if q[2] < 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])   # 180° 绕 x
    axis = axis / na
    ang = np.arccos(np.clip(q @ V0, -1, 1))
    Kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * (Kx @ Kx)


def collect_bouts(K=K_DEFAULT, subset=None, ref="v2"):
    """bout 总表。subset='v12' 排除 data.exclude；ref='v2' 逐鼠垂直零点。

    列：session,start,end,state, gx,gy,gz(raw), grav_std, rx,ry,rz(rotated),
        theta,phi（基于 rotated）。
    """
    ex = set(CFG["data"].get("exclude", [])) if subset == "v12" else set()
    sfx = "_v12" if subset == "v12" else ""
    rows = []
    C = CFG["vame"]["time_window"] // 2
    for s in sessions():
        if s.name in ex:
            continue
        lab = np.load(ROOT / "results" / s.name / f"labels_k{K}{sfx}.npy")
        grav = s.grav[C:C + len(lab)].astype(np.float64)
        R = _align_rotation(np.median(s.grav, axis=0)) if ref == "v2" \
            else np.eye(3)
        grav_r = grav @ R.T
        d = np.diff(lab)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [len(lab)]])
        for a, b in zip(starts, ends):
            if b - a < MIN_BOUT:
                continue
            gm, gr = grav[a:b].mean(0), grav_r[a:b].mean(0)
            gstd = np.linalg.norm(np.diff(grav[a:b], axis=0), axis=1).mean() \
                if b - a > 1 else 0.0
            theta = np.degrees(np.arccos(np.clip(-gr[2], -1, 1)))
            phi = np.degrees(np.arctan2(gr[1], gr[0]))
            rows.append((s.name, int(a), int(b), int(lab[a]),
                         gm[0], gm[1], gm[2], gstd,
                         gr[0], gr[1], gr[2], theta, phi))
    return rows


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fit_buckets(K=K_DEFAULT, subset="v12", ref="v2", kb=4):
    """拟合姿态桶（幂等缓存 cache/buckets_k<K>_v2_k<kb>.npy：
    每 bout 一行 [bucket] 顺序与 posture_bouts 表一致）。"""
    sfx = "_v12" if subset == "v12" else ""
    tag = "_v2" if ref == "v2" else ""
    bp = ROOT / "cache" / f"buckets_k{K}{sfx}{tag}_k{kb}.npy"
    if bp.exists():
        return np.load(bp)
    rows = collect_bouts(K, subset, ref)
    G = np.array([r[8:11] for r in rows])          # rotated grav_mean
    delta = G - V0
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=kb, n_init=10, random_state=0).fit(delta)
    np.save(bp, km.labels_.astype(np.int8))
    np.save(ROOT / "cache" / f"buckets_k{K}{sfx}{tag}_k{kb}_centers.npy",
            km.cluster_centers_)
    return km.labels_


def main(K=K_DEFAULT, subset="v12", ref="v2"):
    sfx = "_v12" if subset == "v12" else ""
    tag = "_v2" if ref == "v2" else ""
    rows = collect_bouts(K, subset, ref)
    with open(ROOT / "cache" / f"posture_bouts_k{K}{sfx}{tag}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "start", "end", "state",
                    "gx", "gy", "gz", "grav_std",
                    "rx", "ry", "rz", "theta", "phi"])
        w.writerows(rows)
    G = np.array([r[8:11] for r in rows])          # rotated grav_mean
    delta = G - V0
    theta = np.array([r[11] for r in rows])
    print(f"[posture{tag}] bout 总数 {len(rows)}（{subset} 口径）；"
          f"θ 中位 {np.median(theta):.1f}° 范围 [{theta.min():.1f}, {theta.max():.1f}]")

    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    pca = PCA(n_components=3).fit(delta)
    Z = pca.transform(delta)
    evr = pca.explained_variance_ratio_
    print(f"[posture{tag}] PCA(δ): PC1={evr[0]*100:.1f}% PC2={evr[1]*100:.1f}% "
          f"PC3={evr[2]*100:.1f}%")

    metrics, labels_k = [], {}
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(delta)
        lb = km.labels_
        labels_k[k] = lb
        sizes = np.bincount(lb) / len(lb)
        metrics.append(dict(k=k, silhouette=round(float(silhouette_score(delta, lb)), 4),
                            inertia=round(float(km.inertia_), 1),
                            min_bucket_pct=round(float(sizes.min() * 100), 1),
                            max_bucket_pct=round(float(sizes.max() * 100), 1)))
        print(f"[posture{tag}] k={k}: sil={metrics[-1]['silhouette']:.3f} "
              f"桶占比 [{sizes.min()*100:.1f}%..{sizes.max()*100:.1f}%]")
    with open(ROOT / "results" / f"posture_k_metrics{tag}.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        w.writerows(metrics)

    plt = _plot_setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]
    ax.scatter(Z[:, 0], Z[:, 1], c=labels_k[4], s=3, cmap="tab10", alpha=0.35)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"姿态流形（{tag} 逐鼠垂直零点，k=4 桶，K={K} {subset}）")
    for i in range(4):
        c = Z[labels_k[4] == i].mean(axis=0)[:2]
        ax.annotate(str(i), c, fontsize=14, fontweight="bold",
                    bbox=dict(boxstyle="circle", fc="white", alpha=0.8))
    ax = axes[1]
    ax.hist(theta, bins=120, color="steelblue")
    km4 = KMeans(n_clusters=4, n_init=10, random_state=0).fit(delta)
    # 桶中心倾角：δ 中心 + v0 → 单位向量
    for i in range(4):
        gv = km4.cluster_centers_[i] + V0
        gv = gv / np.linalg.norm(gv)
        th_c = np.degrees(np.arccos(np.clip(-gv[2], -1, 1)))
        ax.axvline(th_c, color="red", ls="--", alpha=0.6)
        ax.text(th_c, ax.get_ylim()[1] * 0.95, f"桶{i}", color="red")
    ax.set_xlabel("倾角 θ（偏离自己垂直位的角度，度）")
    ax.set_ylabel("bout 数")
    ax.set_title("倾角分布与 k=4 桶中心")
    fig.tight_layout()
    p1 = ROOT / "figures" / f"posture_manifold_k{K}{sfx}{tag}.png"
    fig.savefig(p1, dpi=150)
    print(f"[posture{tag}] -> {p1}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for j, k in enumerate(K_CANDIDATES):
        ax = axes[j]
        ax.scatter(Z[:, 0], Z[:, 1], c=labels_k[k], s=2, cmap="tab10",
                   alpha=0.3)
        ax.set_title(f"k={k}（sil={metrics[j]['silhouette']:.3f}）")
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        if j == 0:
            ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    fig.tight_layout()
    p2 = ROOT / "figures" / f"posture_k_compare{tag}.png"
    fig.savefig(p2, dpi=150)
    print(f"[posture{tag}] -> {p2}")

    sess = np.array([r[0] for r in rows])
    print(f"[posture{tag}] 各 session 在 k=4 下的桶占比（跨鼠可比性检查）:")
    for sn in sorted(set(sess)):
        m = sess == sn
        pct = np.bincount(labels_k[4][m], minlength=4) / m.sum() * 100
        print(f"  {sn}: " + " ".join(f"桶{i}={v:.0f}%" for i, v in enumerate(pct)))
    return metrics


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else K_DEFAULT
    subset = sys.argv[2] if len(sys.argv) > 2 else "v12"
    ref = sys.argv[3] if len(sys.argv) > 3 else "v2"
    main(K, subset, ref)
