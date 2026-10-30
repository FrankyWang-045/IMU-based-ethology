from pathlib import Path
import argparse

import sklearn.utils
from sklearn.utils import check_array as _original_check_array


def _check_array_patched(X, *args, **kwargs):
    force = kwargs.pop("force_all_finite", None)
    if "ensure_all_finite" not in kwargs:
        if force is True:
            kwargs["ensure_all_finite"] = True
        elif force is False:
            kwargs["ensure_all_finite"] = "allow-nan"
        elif force is not None:
            kwargs["ensure_all_finite"] = force
    return _original_check_array(X, *args, **kwargs)


sklearn.utils.check_array = _check_array_patched

import matplotlib.pyplot as plt
import numpy as np
import umap
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries

from utils import load_config, output_dir


class DSU:
    """并查集，用于 persistence 合并树。"""

    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def coords_to_grid(coords, xedges, yedges, bins):
    ix = np.digitize(coords[:, 0], xedges) - 1
    iy = np.digitize(coords[:, 1], yedges) - 1
    return np.clip(ix, 0, bins - 1), np.clip(iy, 0, bins - 1)


def compute_umap(segment, segment_time, n_neighbors=15, min_dist=0.1, metric="euclidean", time_weight=0.0):
    if time_weight > 0:
        t_norm = (segment_time - segment_time.min()) / (segment_time.max() - segment_time.min() + 1e-8)
        segment_for_umap = np.hstack([segment, t_norm[:, None] * time_weight])
    else:
        segment_for_umap = segment

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(segment_for_umap)


def detect_peaks(H_smooth, min_distance=3, threshold_rel=0.05, exclude_border=2):
    peaks = peak_local_max(
        H_smooth,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
    )
    return peaks


def compute_persistence_tree(H_smooth):
    h, w = H_smooth.shape

    peaks = detect_peaks(H_smooth)
    n_peaks = len(peaks)
    if n_peaks == 0:
        return peaks, np.array([]), []

    peak_cells = peaks[:, 0] * w + peaks[:, 1]
    peak_densities = H_smooth[peaks[:, 0], peaks[:, 1]]

    order = np.argsort(H_smooth.ravel())[::-1]

    dsu = DSU(h * w)
    active = np.zeros(h * w, dtype=bool)
    comp_peak = np.full(h * w, -1, dtype=int)
    comp_peak_density = np.zeros(h * w)

    for i, pc in enumerate(peak_cells):
        comp_peak[pc] = i
        comp_peak_density[pc] = peak_densities[i]

    merge_events = []

    for cell in order:
        active[cell] = True
        r, c = divmod(cell, w)

        neighbors = []
        if r > 0:
            neighbors.append(cell - w)
        if r < h - 1:
            neighbors.append(cell + w)
        if c > 0:
            neighbors.append(cell - 1)
        if c < w - 1:
            neighbors.append(cell + 1)

        active_neighbors = [n for n in neighbors if active[n]]

        if not active_neighbors:
            if comp_peak[cell] == -1:
                comp_peak[cell] = n_peaks
                comp_peak_density[cell] = H_smooth.ravel()[cell]
                n_peaks += 1
            continue

        neighbor_comps = set()
        for nb in active_neighbors:
            neighbor_comps.add(dsu.find(nb))

        if len(neighbor_comps) == 1:
            comp = neighbor_comps.pop()
            dsu.parent[cell] = comp
        else:
            best_comp = max(neighbor_comps, key=lambda x: comp_peak_density[x])
            for comp in neighbor_comps:
                if comp != best_comp:
                    dsu.union(comp, best_comp)
                    dying_peak = comp_peak[comp]
                    surviving_peak = comp_peak[best_comp]
                    if dying_peak != -1:
                        merge_density = H_smooth.ravel()[cell]
                        merge_events.append((merge_density, dying_peak, surviving_peak))
            dsu.parent[cell] = best_comp

    merge_events.sort(key=lambda x: x[0])
    return peaks, peak_densities, merge_events


def get_surviving_peaks(peaks, peak_densities, merge_events, persistence_thr_ratio):
    n_peaks = len(peaks)
    parent = np.arange(n_peaks)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for merge_density, dying, surviving in merge_events:
        if dying < 0 or dying >= n_peaks or surviving < 0 or surviving >= n_peaks:
            continue
        peak_density = peak_densities[dying]
        if peak_density <= 0:
            continue
        persistence_ratio = (peak_density - merge_density) / peak_density
        if persistence_ratio < persistence_thr_ratio:
            parent[root(dying)] = root(surviving)

    survivors = set()
    for i in range(n_peaks):
        if root(i) == i:
            survivors.add(i)
    return list(survivors)


def compute_basins(H_smooth, peaks, survivors):
    markers = np.zeros_like(H_smooth, dtype=int)
    for new_id, peak_idx in enumerate(survivors, start=1):
        r, c = peaks[peak_idx]
        markers[r, c] = new_id
    labels = watershed(-H_smooth, markers=markers)
    return labels


def hierarchical_peak_watershed(
    cfg,
    recording_name,
    latent_subdir="latent",
    bins=200,
    sigma=2.0,
    n_levels=10,
    persistence_max=0.5,
    persistence_min=0.02,
    min_distance=3,
    threshold_rel=0.05,
    exclude_border=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    time_weight=0.0,
    max_points=None,
    cmap="tab20",
    dpi=100,
):
    out = output_dir(cfg)
    latent_dir = out / latent_subdir
    npz_path = latent_dir / f"{recording_name}_latent.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 latent: {npz_path}")

    data = np.load(npz_path)
    latent = data["latent"]
    time = data["time"]

    segment = latent
    segment_time = time

    if max_points is not None and len(segment) > max_points:
        idx = np.linspace(0, len(segment) - 1, max_points, dtype=int)
        segment = segment[idx]
        segment_time = segment_time[idx]
        print(f"  点数超过 {max_points}，已均匀降采样至 {max_points}")

    print("UMAP 降维...")
    coords = compute_umap(
        segment,
        segment_time,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        time_weight=time_weight,
    )

    print("构建密度图...")
    H, xedges, yedges = np.histogram2d(coords[:, 0], coords[:, 1], bins=bins)
    H_smooth = gaussian_filter(H, sigma=sigma)
    H_norm = H_smooth / (H_smooth.max() + 1e-12)

    print("检测山峰并计算 persistence...")
    peaks, peak_densities, merge_events = compute_persistence_tree(H_smooth)
    print(f"  共检测到 {len(peaks)} 个局部极大值")

    if len(peaks) == 0:
        print("未检测到任何山峰，退出")
        return

    ratios = np.linspace(0, 1, n_levels) ** 2
    thresholds = persistence_max - (persistence_max - persistence_min) * ratios

    labels_levels = []
    survivors_levels = []

    print("分层 watershed...")
    for level, thr in enumerate(thresholds):
        survivors = get_surviving_peaks(peaks, peak_densities, merge_events, thr)
        labels = compute_basins(H_smooth, peaks, survivors)
        labels_levels.append(labels)
        survivors_levels.append(survivors)
        print(f"  level {level:2d}: persistence_thr={thr:.3f}, peaks={len(survivors)}, states={labels.max()}")

    # 修正：labels 的第一维是 y，第二维是 x，因此用 labels[iy, ix]
    ix, iy = coords_to_grid(coords, xedges, yedges, bins)
    state_levels = [labels[iy, ix].astype(np.int32) for labels in labels_levels]

    save_path = latent_dir / f"{recording_name}_peak_hierarchical_states.npz"
    np.savez_compressed(
        save_path,
        time=segment_time,
        latent=segment.astype(np.float32),
        coords=coords.astype(np.float32),
        thresholds=thresholds,
        labels_levels=np.stack(labels_levels, axis=0).astype(np.int32),
        state_levels=np.stack(state_levels, axis=0).astype(np.int32),
        peaks=peaks.astype(np.int32),
        peak_densities=peak_densities.astype(np.float32),
        xedges=xedges,
        yedges=yedges,
    )
    print(f"已保存 {save_path}")

    cols = 3
    rows = (n_levels + cols - 1) // cols
    figsize = (1920 / dpi, 1080 / dpi)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes).flatten()

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    for level in range(n_levels):
        ax = axes[level]
        labels_grid = labels_levels[level]
        survivors = survivors_levels[level]

        # 密度背景：不再转置
        ax.imshow(H_norm, origin="lower", extent=extent, cmap="gray", alpha=0.5, aspect="auto")

        # basin 上色，0 为透明
        labels_vis = labels_grid.astype(float)
        labels_vis[labels_vis == 0] = np.nan
        vmax = max(1, int(labels_grid.max()))
        ax.imshow(
            labels_vis,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            alpha=0.8,
            aspect="auto",
        )

        # 山脊边界线：不再转置
        boundaries = find_boundaries(labels_grid, mode="thick")
        boundary_overlay = np.zeros((*boundaries.shape, 4))
        boundary_overlay[boundaries] = [0, 0, 0, 0.6]
        ax.imshow(boundary_overlay, origin="lower", extent=extent, aspect="auto")

        # 标出存活的山峰
        for peak_idx in survivors:
            r, c = peaks[peak_idx]
            ax.plot(
                xedges[c] + (xedges[1] - xedges[0]) / 2,
                yedges[r] + (yedges[1] - yedges[0]) / 2,
                "o",
                color="white",
                markeredgecolor="black",
                markersize=5,
                markeredgewidth=0.5,
            )

        ax.set_title(
            f"level {level} | persist={thresholds[level]:.2f} | peaks={len(survivors)}",
            fontsize=8,
        )
        ax.set_xlabel("UMAP 1", fontsize=7)
        ax.set_ylabel("UMAP 2", fontsize=7)
        ax.tick_params(labelsize=6)

    for idx in range(n_levels, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout(pad=0.5)
    vis_path = latent_dir / f"{recording_name}_peak_hierarchical_watershed.png"
    plt.savefig(vis_path, dpi=dpi, bbox_inches="tight")
    print(f"可视化已保存: {vis_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="基于 persistence 的山峰-山脊分层 watershed")
    parser.add_argument("--recording", required=True)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--n_levels", type=int, default=10)
    parser.add_argument("--persistence_max", type=float, default=0.5)
    parser.add_argument("--persistence_min", type=float, default=0.02)
    parser.add_argument("--min_distance", type=int, default=3)
    parser.add_argument("--threshold_rel", type=float, default=0.05)
    parser.add_argument("--exclude_border", type=int, default=2)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--time_weight", type=float, default=0.0)
    parser.add_argument("--max_points", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    hierarchical_peak_watershed(
        cfg,
        args.recording,
        bins=args.bins,
        sigma=args.sigma,
        n_levels=args.n_levels,
        persistence_max=args.persistence_max,
        persistence_min=args.persistence_min,
        min_distance=args.min_distance,
        threshold_rel=args.threshold_rel,
        exclude_border=args.exclude_border,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        time_weight=args.time_weight,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()