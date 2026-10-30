"""
Multi-level watershed visualization: point-based rendering with alpha shapes.

Figure 1 (xxx_ws_levels.png):
  - panel 0: 2D display-density terrain (context only)
  - one panel per level: submerged frames in light gray, above-water frames
    colored by island, per-island boundary drawn as an alpha shape computed
    from the island's own points (no grid voting, no hole filling), island
    centers marked with x + ID

Figure 2 (xxx_ws_timelines.png): one state timeline strip per level
(sea = -1 shown in light gray)

Usage:
  python plot_ws_hierarchy.py --hier_path xxx_ws_hierarchy.npz
"""
import argparse
import os
import sys
import time
from pathlib import Path

print(sys.executable)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree, Delaunay


def save_fig(fig, path, dpi=150, retries=3):
    path = os.path.normpath(path)
    for attempt in range(retries):
        try:
            fig.savefig(path, dpi=dpi)
            print(f"Saved: {path}")
            return
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"save failed ({e}), retry {attempt + 1}/{retries} ...")
            time.sleep(1)


def load_or_make_embedding(latent_dir, recording, latent):
    """
    统一坐标系嵌入（与 visualize_umap_terrain.py 共用）：
    UMAP(n_neighbors=15, min_dist=0.1, random_state=42)，
    输入为与 islands npz 对齐的 [::4] 分析帧。
    首次运行计算并保存 <recording>_embedding.npy，之后所有脚本复用，
    保证各图坐标系完全一致。
    """
    p = Path(latent_dir) / f"{recording}_embedding.npy"
    if p.exists():
        print(f"使用共享嵌入: {p}")
        return np.load(p)
    import umap
    print("首次计算共享嵌入 UMAP(15, 0.1, rs42) ...")
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    random_state=42).fit_transform(latent)
    np.save(p, emb)
    return emb


def density_grid(emb, res=200, sigma=2):
    mn, mx = emb.min(axis=0), emb.max(axis=0)
    pad = 0.05 * (mx - mn)
    H, xedges, yedges = np.histogram2d(
        emb[:, 0], emb[:, 1], bins=res,
        range=[[mn[0] - pad[0], mx[0] + pad[0]],
               [mn[1] - pad[1], mx[1] + pad[1]]])
    # numpy histogram2d 输出轴序为 [x, y]（第一轴=x 分箱），而 contour
    # 系函数假设第一轴为行(y)。转置为 [y, x] 后与手构数组布局一致。
    D = gaussian_filter(H.T, sigma=sigma)
    D /= max(D.max(), 1e-12)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return D, extent


def alpha_shape_segments(pts, alpha_mult=3.0, max_pts=2000, seed=0):
    """
    从点集计算 alpha-shape 边界线段集合。
    返回 (k, 2, 2) 的线段数组；点数不足或退化时返回 None。
    alpha_mult: 边界松弛倍数，越大轮廓越凸（趋向凸包），越小越贴合。
    """
    if len(pts) < 4:
        return None
    if len(pts) > max_pts:
        idx = np.random.default_rng(seed).choice(
            len(pts), max_pts, replace=False)
        pts = pts[idx]
    pts = np.unique(pts, axis=0)
    if len(pts) < 4:
        return None
    nn, _ = cKDTree(pts).query(pts, k=2)
    R = alpha_mult * np.median(nn[:, 1])
    try:
        tri = Delaunay(pts)
    except Exception:
        return None
    p0, p1, p2 = pts[tri.simplices[:, 0]], pts[tri.simplices[:, 1]], \
        pts[tri.simplices[:, 2]]
    a = np.linalg.norm(p0 - p1, axis=1)
    b = np.linalg.norm(p1 - p2, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 1e-24))
    circum = a * b * c / (4.0 * area)
    keep = circum <= R
    if not keep.any():
        return None
    edges = tri.simplices[keep][:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return None
    return pts[bnd]


def main():
    parser = argparse.ArgumentParser(description="Watershed hierarchy viz")
    parser.add_argument("--hier_path", required=True,
                        help="*_ws_hierarchy.npz (or *_islands_hierarchy.npz)")
    parser.add_argument("--out_prefix",
                        help="output prefix (default: strip _hierarchy)")
    parser.add_argument("--max_points", type=int, default=40000,
                        help="scatter subsample cap per level")
    parser.add_argument("--max_seg_lines", type=int, default=800,
                        help="max segments drawn per timeline strip")
    parser.add_argument("--alpha_mult", type=float, default=8.0,
                        help="alpha-shape looseness; larger = more convex")
    parser.add_argument("--context_alpha", type=float, default=0.18,
                        help="alpha of submerged (sea) context points")
    args = parser.parse_args()

    hier = Path(args.hier_path)
    d = np.load(hier)
    latent = d["latent"]
    time_ds = d["time"] if "time" in d else np.arange(len(latent))
    h_levels = d["h_levels"]
    n_per_level = d["n_states_per_level"]
    frame_density = d["frame_density"]
    L = len(h_levels)

    out_prefix = args.out_prefix or args.hier_path.replace(
        "_hierarchy.npz", "")

    # 统一坐标系：与 visualize_umap_terrain.py 共用同一嵌入文件
    recording = hier.name
    for suffix in ("_islands_hierarchy.npz", "_ws_hierarchy.npz",
                   "_hierarchy.npz"):
        if recording.endswith(suffix):
            recording = recording[:-len(suffix)]
            break
    print("Embedding (shared with visualize_umap_terrain.py)...")
    emb = load_or_make_embedding(hier.parent, recording, latent)
    D2d, extent = density_grid(emb)

    sel = np.arange(len(emb))
    if len(sel) > args.max_points:
        sel = np.random.default_rng(0).choice(len(emb), args.max_points,
                                              replace=False)
    emb_s = emb[sel]
    fd_s = frame_density[sel]

    cmap = plt.get_cmap("tab20")

    # ============ Figure 1: terrain + per-level point/alpha-shape maps ===
    n_panel = L + 1
    ncols = 2 if n_panel > 1 else 1
    nrows = (n_panel + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    ax = axes[0]
    cf = ax.contourf(D2d, extent=extent, levels=20, cmap="terrain",
                     alpha=0.9)
    fig.colorbar(cf, ax=ax, label="density")
    ax.set_title("Density terrain (display only; shared background)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    for li in range(L):
        ax = axes[li + 1]
        labels = d[f"states_L{li}"]
        n_states = int(n_per_level[li])
        h = float(h_levels[li])
        above = fd_s >= h

        # 背景地形等高线（淡）
        ax.contour(D2d, extent=extent, levels=8, colors="gray",
                   linewidths=0.3, alpha=0.35)

        # 水下帧：浅灰上下文（透明度要足够看到完整点云轮廓，否则会
        # 误以为背景地形与点云错位）
        ax.scatter(emb_s[~above, 0], emb_s[~above, 1], s=1, c="0.75",
                   alpha=args.context_alpha, linewidths=0)

        # 露出水面的帧：按岛着色（底层细点）
        vmax_colors = max(19, n_states)
        ax.scatter(emb_s[above, 0], emb_s[above, 1],
                   c=labels[sel][above], cmap=cmap, vmin=0,
                   vmax=vmax_colors - 1, s=2, alpha=0.4, linewidths=0)

        # 每个岛的 alpha-shape 边界（从岛自身的点计算）
        above_full = frame_density >= h
        counts_above = np.bincount(labels[above_full], minlength=n_states)
        for u in range(n_states):
            mask = (labels == u) & above_full
            if counts_above[u] == 0 or not mask.any():
                continue
            color_u = cmap(u / max(vmax_colors - 1, 1))
            pts_u = emb[mask]
            segs = alpha_shape_segments(pts_u,
                                        alpha_mult=args.alpha_mult)
            if segs is not None:
                # 边界用岛自身颜色加深绘制
                ax.add_collection(LineCollection(
                    segs, colors=[color_u], linewidths=1.4, alpha=0.95))
            # 小岛把点画大画实，避免被主导岛淹没
            small = counts_above[u] < 0.05 * max(total_above := int(
                above_full.sum()), 1)
            m_s = mask[sel]
            if m_s.any():
                ax.scatter(emb_s[m_s, 0], emb_s[m_s, 1], s=(8 if small else 3),
                           c=[color_u], alpha=0.95, linewidths=0,
                           zorder=5 if small else 3)
            cx, cy = pts_u.mean(axis=0)
            ax.plot(cx, cy, marker="x", color="black", markersize=10,
                    markeredgewidth=2, zorder=6)
            ax.text(cx, cy, f" {u}", fontsize=11, fontweight="bold",
                    ha="left", va="center", zorder=6,
                    bbox=dict(facecolor="white", alpha=0.8,
                              edgecolor="black", pad=1.5))

        total_above = int(above_full.sum())
        top = np.argsort(-counts_above)[:5]
        legend_txt = "Top-5 above-water share:\n" + "\n".join(
            f"island {u}: {counts_above[u]/max(total_above,1)*100:.1f}%"
            for u in top)
        ax.text(0.98, 0.98, legend_txt, transform=ax.transAxes,
                fontsize=9, va="top", ha="right",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray"))
        ax.set_title(f"L{li}: h={h:g}  ({n_states} islands, "
                     f"{total_above/len(labels)*100:.0f}% of frames "
                     "above water)")
        ax.set_xlabel("embedding dim 1")
        ax.set_ylabel("embedding dim 2")
        ax.set_aspect("equal")
        # 固定坐标范围 = 背景地形 extent，避免各面板 autoscale 不一致
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    for j in range(n_panel, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_fig(fig, out_prefix + "_levels.png")
    plt.close(fig)

    # ============ Figure 2: per-level state timelines ============
    fig, axes = plt.subplots(L, 1, figsize=(14, 1.8 * L), sharex=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)

    for li in range(L):
        ax = axes[li]
        labels = d[f"states_L{li}"]
        starts = d[f"segment_starts_L{li}"]
        lengths = d[f"segment_lengths_L{li}"]
        sel_seg = np.arange(len(starts))
        if len(sel_seg) > args.max_seg_lines:
            sel_seg = rng.choice(len(starts), args.max_seg_lines,
                                 replace=False)
        vmax_l = max(1, int(n_per_level[li]) - 1)
        for si in sel_seg:
            s, ln = int(starts[si]), int(lengths[si])
            e = min(s + ln - 1, len(time_ds) - 1)
            color = ("lightgray" if labels[s] < 0
                     else cmap(labels[s] / vmax_l))
            ax.hlines(labels[s], time_ds[s], time_ds[e],
                      colors=color, linewidth=2)
        ax.set_ylabel(f"L{li}\n{int(n_per_level[li])} basins", fontsize=9)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Per-level state timelines (segment subsample)", y=1.0)
    plt.tight_layout()
    save_fig(fig, out_prefix + "_timelines.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
