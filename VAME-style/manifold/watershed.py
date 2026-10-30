"""
泛洪/分水岭行为分割（超水平集持久性版）。

原理：把 latent（PCA 前 m 维）的 KDE 密度看作地形，从高处向低处
泛洪（union-find 合并）。两个峰在鞍部合并时的落差 = 持久性。
持久性 >= h 的峰保留为独立盆地；其余峰被并入杀死它的峰。

参数只有一个 h（相对持久性阈值），不需要预先定义状态数。

用法：
    python segment_watershed.py --latent_path xxx_latent.npz
        （未给 --h_persistence 时先打印 h-盆地数扫描表，供选 h）
    python segment_watershed.py --latent_path xxx_latent.npz --h_persistence 0.1
"""
import argparse
import sys
import time
from pathlib import Path

print(sys.executable)

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, label as ndi_label
from scipy.spatial import cKDTree

# ---------- PCA（只做正交投影，不改度量） ----------
def pca_project(X, variance=0.9, max_dim=4, min_dim=2):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    cvar = np.cumsum(var) / var.sum()
    m = int(np.searchsorted(cvar, variance)) + 1
    m = max(min_dim, min(m, max_dim, X.shape[1]))
    print(f"PCA: 前 {m} 维解释方差 {cvar[m-1]*100:.1f}%")
    return U[:, :m] * S[:m], m

# ---------- KDE 网格 ----------
def kde_grid(Y, res, sigma_cells):
    m = Y.shape[1]
    mins = Y.min(axis=0)
    maxs = Y.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    edges = [np.linspace(0, 1, res + 1) for _ in range(m)]
    Z = (Y - mins) / span  # [0,1]^m
    H, _ = np.histogramdd(Z, bins=edges)
    D = gaussian_filter(H, sigma=sigma_cells, mode="nearest")
    D /= D.max()
    return D, mins, span, Z

# ---------- 泛洪：峰检测 + 持久性 ----------
def flood_persistence(D):
    """
    返回:
      peaks_flat: 每个峰代表单元的扁平索引
      peak_h:     峰顶密度
      persist:    每个峰的持久性（全局最高峰 = inf）
      killer:     每个峰死亡时并入的峰索引（指向 peaks_flat 下标）
      cell_peak:  每个单元所属组件的存活峰（peaks_flat 下标，-1 表示无密度）
    """
    m = D.ndim
    N = D.size
    d = D.ravel()

    # 局部极大值平板（连通等值区）作为候选峰
    from scipy.ndimage import generate_binary_structure
    struct = generate_binary_structure(m, 1)
    maxf = maximum_filter(D, footprint=struct, mode="nearest")
    pm, _ = ndi_label((D == maxf) & (D > 0), structure=struct)
    n_peaks = pm.max()
    peak_of_cell = np.full(N, -1, dtype=np.int64)   # 非峰单元 = -1
    peaks_flat = np.zeros(n_peaks, dtype=np.int64)
    peak_h = np.zeros(n_peaks)
    for p in range(1, n_peaks + 1):
        cells = np.flatnonzero(pm.ravel() == p)
        c = cells[np.argmax(d[cells])]
        peaks_flat[p - 1] = c
        peak_h[p - 1] = d[c]
        peak_of_cell[cells] = p - 1

    # union-find，密度降序扫描
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # 组件根 -> 存活峰下标；非峰单元为 -1（组件尚未包含峰）
    root_peak = peak_of_cell.copy()
    persist = np.full(n_peaks, np.inf)
    killer = np.full(n_peaks, -1, dtype=np.int64)

    order = np.argsort(-d)
    processed = np.zeros(N, dtype=bool)
    for c in order:
        if d[c] <= 0:
            break
        processed[c] = True
        idx = np.unravel_index(c, D.shape)
        for axis in range(m):
            for step in (-1, 1):
                nb = list(idx)
                nb[axis] += step
                if nb[axis] < 0 or nb[axis] >= D.shape[axis]:
                    continue
                n = np.ravel_multi_index(nb, D.shape)
                if not processed[n]:
                    continue
                ra, rb = find(c), find(n)
                if ra == rb:
                    continue
                pa, pb = root_peak[ra], root_peak[rb]
                # 组件可能还没有峰（-1）：空峰的高度视为 -inf
                ha = peak_h[pa] if pa >= 0 else -np.inf
                hb = peak_h[pb] if pb >= 0 else -np.inf
                if ha >= hb:
                    parent[rb] = ra
                    root = ra
                    if pb >= 0:          # 杀死的组件里有峰才记录持久性
                        dead, surv = pb, (pa if pa >= 0 else pb)
                        persist[dead] = peak_h[dead] - d[c]
                        killer[dead] = surv
                        root_peak[root] = surv
                    else:
                        root_peak[root] = pa
                else:
                    parent[ra] = rb
                    root = rb
                    dead, surv = pa, pb
                    persist[dead] = peak_h[dead] - d[c]
                    killer[dead] = surv
                    root_peak[root] = surv

    # 每个单元的最终存活峰
    cell_peak = np.full(N, -1, dtype=np.int64)
    for c in order:
        if d[c] <= 0:
            break
        cell_peak[c] = root_peak[find(c)]

    return peaks_flat, peak_h, persist, killer, cell_peak

def resolve_labels(cell_peak, persist, killer, h):
    """按持久性阈值 h 把被杀死的峰逐级重定向到最终保留的峰。"""
    final = np.full(len(persist), -1, dtype=np.int64)

    def resolve(p):
        while persist[p] < h:
            p = killer[p]
        return p

    keep = np.array([p for p in range(len(persist))
                     if persist[p] >= h], dtype=np.int64)
    remap = {p: i for i, p in enumerate(
        sorted(keep, key=lambda p: -persist[p]))}
    out = np.full(len(cell_peak), -1, dtype=np.int64)
    for c in np.flatnonzero(cell_peak >= 0):
        out[c] = remap[resolve(cell_peak[c])]
    return out, len(keep)

# ---------- 帧标签 ----------
def assign_frames(Z, D_shape, cell_labels):
    idx = np.clip((Z * D_shape[0]).astype(int), 0, D_shape[0] - 1)
    cells = np.ravel_multi_index(idx.T, D_shape)
    labels = cell_labels[cells]
    # 密度为零区域的单元标签为 -1，就近补齐
    missing = labels < 0
    if missing.any():
        occ = np.flatnonzero(cell_labels >= 0)
        centers = np.array(np.unravel_index(occ, D_shape)).T
        tree = cKDTree(centers)
        _, j = tree.query(idx[missing], k=1)
        labels[missing] = cell_labels[occ[j]]
    return labels

def get_segments(labels):
    bounds = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    return bounds[:-1], np.diff(bounds)

def enforce_min_duration(labels, min_len, max_iter=1000):
    labels = labels.copy()
    for _ in range(max_iter):
        starts, lengths = get_segments(labels)
        short = np.flatnonzero(lengths < min_len)
        if len(short) == 0:
            break
        for si in short[np.argsort(lengths[short])]:
            s, L = starts[si], lengths[si]
            e = s + L
            left = labels[s - 1] if s > 0 else None
            right = labels[e] if e < len(labels) else None
            if left is None and right is None:
                break
            if left is None:
                labels[s:e] = right
            elif right is None:
                labels[s:e] = left
            else:
                ll = lengths[si - 1] if si > 0 else 0
                rl = lengths[si + 1] if si + 1 < len(lengths) else 0
                labels[s:e] = left if ll >= rl else right
    return labels

def main():
    parser = argparse.ArgumentParser(
        description="泛洪/分水岭行为分割（持久性阈值，无需预定义状态数）"
    )
    parser.add_argument("--recording", help="recording 名（与 utils 配置一起用）")
    parser.add_argument("--latent_path", help="直接指定 *_latent.npz")
    parser.add_argument("--out_path", help="输出 npz 路径")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--min_duration", type=int, default=10,
                        help="最短段长，原始 100 Hz 帧")
    parser.add_argument("--h_persistence", type=float, default=None,
                        help="持久性阈值(0~1)；不给则只打印扫描表")
    parser.add_argument("--variance", type=float, default=0.9,
                        help="PCA 保留累积方差比例")
    parser.add_argument("--res", type=int, default=0,
                        help="每维网格数（默认按维度自适应）")
    parser.add_argument("--kde_sigma", type=float, default=2.0,
                        help="KDE 带宽，单位为网格数")
    args = parser.parse_args()

    if args.latent_path is None and args.recording is None:
        parser.error("需要 --recording 或 --latent_path 之一")

    if args.latent_path is not None:
        npz_path = args.latent_path
    else:
        from utils import load_config, output_dir
        latent_dir = output_dir(load_config()) / "latent"
        npz_path = str(latent_dir / f"{args.recording}_latent.npz")

    if not Path(npz_path).exists():
        raise FileNotFoundError(f"未找到 latent: {npz_path}")

    data = np.load(npz_path)
    latent = data["latent"].astype(np.float64)
    time_full = data["time"] if "time" in data else np.arange(len(latent))

    step = args.downsample
    latent = np.ascontiguousarray(latent[::step])
    time_ds = time_full[::step]

    scale = latent.std(axis=0)
    scale[scale == 0] = 1.0
    X = latent / scale

    t0 = time.perf_counter()
    Y, m = pca_project(X, variance=args.variance)
    res = args.res or {2: 200, 3: 100, 4: 48}.get(m, 48)
    D, mins, span, Z = kde_grid(Y, res, args.kde_sigma)
    print(f"网格: {res}^{m} = {D.size:,} 格, KDE+泛洪中...")

    peaks_flat, peak_h, persist, killer, cell_peak = flood_persistence(D)
    print(f"候选峰: {len(peaks_flat)} 个, 耗时 {time.perf_counter()-t0:.1f}s")

    # ---- 模式一：扫描表 ----
    if args.h_persistence is None:
        print("\nh 扫描表（h = 允许峰间鞍部相对落差）:")
        print(f"{'h':>8} {'盆地数':>8}")
        for h in (0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3):
            _, n = resolve_labels(cell_peak, persist, killer, h)
            print(f"{h:8.3f} {n:8d}")
        top = np.argsort(-np.where(np.isinf(persist), peak_h, persist))[:30]
        print("\ntop-30 峰（密度 / 持久性）:")
        for p in top:
            ps = "inf" if np.isinf(persist[p]) else f"{persist[p]:.4f}"
            print(f"  peak {p:5d}: height={peak_h[p]:.4f} persistence={ps}")
        print("\n请根据扫描表 knee 处选 h 重跑: --h_persistence <值>")
        return

    # ---- 模式二：正式分割 ----
    cell_labels, n_keep = resolve_labels(
        cell_peak, persist, killer, args.h_persistence)
    print(f"保留盆地: {n_keep} 个 (h={args.h_persistence})")

    labels = assign_frames(Z, D.shape, cell_labels)
    min_len = max(2, int(round(args.min_duration / step)))
    labels = enforce_min_duration(labels, min_len)
    starts, lengths = get_segments(labels)

    uniq = np.unique(labels)
    remap = np.empty(uniq.max() + 1, dtype=np.int64)
    remap[uniq] = np.arange(len(uniq))
    labels = remap[labels]

    centers = np.array([X[labels == u].mean(axis=0)
                        for u in np.unique(labels)]) * scale

    out_path = args.out_path
    if out_path is None:
        p = Path(npz_path)
        rec = args.recording or p.name.replace("_latent.npz", "")
        out_path = str(p.parent / f"{rec}_ws_states.npz")

    np.savez_compressed(
        out_path,
        states=labels.astype(np.int32),
        time=time_ds,
        latent=latent.astype(np.float32),
        centers=centers.astype(np.float32),
        segment_starts=starts.astype(np.int32),
        segment_lengths=lengths.astype(np.int32),
        h_persistence=args.h_persistence,
        n_states=len(np.unique(labels)),
        downsample=step,
        min_duration=args.min_duration,
        pca_dim=m,
        pca_variance=args.variance,
        kde_sigma=args.kde_sigma,
    )
    print(f"结果已保存: {out_path}")
    print(f"  有效模态数: {len(np.unique(labels))}")
    print(f"  片段数: {len(lengths)}")
    print(f"  最短段长: {lengths.min()} 帧（降采样后）")
    print(f"  中位段长: {np.median(lengths):.1f} 帧（降采样后）")

if __name__ == "__main__":
    main()