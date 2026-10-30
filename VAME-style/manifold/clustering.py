"""
Density-peaks 聚类（Rodriguez & Laio 2014）+ 最小段长后处理的行为分割。

输入：AE latent（T × D，100 Hz）
输出：每帧的模态标签 + 段序列

用法（与你的 utils/config 集成）：
    python segment_density_peaks.py --recording A1_C1_A1-6d --n_states 40

用法（直接指定文件，不依赖 utils）：
    python segment_density_peaks.py --latent_path path/to/xxx_latent.npz --n_states 40
"""
import argparse
import sys
import time
from pathlib import Path

print(sys.executable)

import numpy as np
from scipy.spatial import cKDTree

def compute_rho_delta(X, k=10):
    """
    局部密度 rho 与到最近高密度点的距离 delta（fast-DPC 近似，O(N·k)）。
    对没有更高密度近邻的点（候选峰），delta 取 kNN 最大距离。
    """
    N = len(X)
    tree = cKDTree(X)
    dists, idxs = tree.query(X, k=k + 1)
    dists = dists[:, 1:]          # 去掉自身
    idxs = idxs[:, 1:]

    dc = np.median(dists[:, -1])
    if dc <= 0:
        dc = X.std() + 1e-12
    rho = np.exp(-(dists / dc) ** 2).sum(axis=1)

    delta = np.empty(N)
    nn_higher = np.empty(N, dtype=np.int64)

    order = np.argsort(-rho)
    rank = np.empty(N, dtype=np.int64)
    rank[order] = np.arange(N)

    for i in range(N):
        mask = rank[idxs[i]] < rank[i]
        if mask.any():
            d_local = dists[i][mask]
            j = idxs[i][mask][np.argmin(d_local)]
            delta[i] = np.linalg.norm(X[i] - X[j])
            nn_higher[i] = j
        else:
            delta[i] = dists[i].max()
            nn_higher[i] = i

    return rho, delta, nn_higher
def density_peaks_cluster(X, n_states=0, k=10, auto_sigma=2.0):
    """
    返回标签（按占用频次排序）、峰索引、rho、delta。
    n_states > 0：取 gamma 最大的 n_states 个点为峰；
    n_states = 0：自动阈值 mean + auto_sigma * std。
    """
    N = len(X)
    rho, delta, nn_higher = compute_rho_delta(X, k=k)
    gamma = rho * delta

    if n_states > 0:
        n_states = min(n_states, N)
        peaks = np.argsort(-gamma)[:n_states]
    else:
        thresh = gamma.mean() + auto_sigma * gamma.std()
        peaks = np.flatnonzero(gamma > thresh)
        if len(peaks) == 0:
            raise RuntimeError(
                "自动阈值未找到峰，请降低 --auto_sigma 或显式指定 --n_states"
            )

    is_peak = np.zeros(N, dtype=bool)
    is_peak[peaks] = True

    # 密度降序分配：最近更高密度点一定先被处理。
    # 用 -1 做哨兵；kNN 近似下未入选的候选点会自引用，标签保持 -1，
    # 循环结束后统一兜底。
    labels = np.full(N, -1, dtype=np.int64)
    labels[peaks] = np.arange(len(peaks))
    for i in np.argsort(-rho):
        if not is_peak[i]:
            labels[i] = labels[nn_higher[i]]

    # 兜底：把未被分配的点（自引用的未入选候选峰）指给最近的有标签点
    unassigned = labels < 0
    if unassigned.any():
        assigned_idx = np.flatnonzero(~unassigned)
        tree2 = cKDTree(X[assigned_idx])
        _, j = tree2.query(X[unassigned], k=1)
        labels[unassigned] = labels[assigned_idx[j]]

    counts = np.bincount(labels, minlength=len(peaks))
    remap = np.empty(len(peaks), dtype=np.int64)
    remap[np.argsort(-counts)] = np.arange(len(peaks))
    return remap[labels], peaks, rho, delta

def get_segments(labels):
    bounds = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    return bounds[:-1], np.diff(bounds)

def enforce_min_duration(labels, min_len, max_iter=1000):
    """短于 min_len 的段反复并入较长的邻居，直到全部满足约束。"""
    labels = labels.copy()
    for _ in range(max_iter):
        starts, lengths = get_segments(labels)
        short = np.flatnonzero(lengths < min_len)
        if len(short) == 0:
            break
        for si in short[np.argsort(lengths[short])]:
            s, L = starts[si], lengths[si]
            e = s + L
            left_label = labels[s - 1] if s > 0 else None
            right_label = labels[e] if e < len(labels) else None
            if left_label is None and right_label is None:
                break
            if left_label is None:
                labels[s:e] = right_label
            elif right_label is None:
                labels[s:e] = left_label
            else:
                left_len = lengths[si - 1] if si > 0 else 0
                right_len = lengths[si + 1] if si + 1 < len(lengths) else 0
                labels[s:e] = left_label if left_len >= right_len else right_label
    return labels

def main():
    parser = argparse.ArgumentParser(
        description="Density-peaks 行为分割（空间聚类 + 最小段长后处理）"
    )
    parser.add_argument("--recording", help="recording 名（与 utils 配置一起用）")
    parser.add_argument("--latent_path", help="直接指定 *_latent.npz，跳过 utils 配置")
    parser.add_argument("--out_path", help="输出 npz 路径（默认写在 latent 同目录）")
    parser.add_argument("--downsample", type=int, default=4,
                        help="降采样因子，4 = 100 Hz 降到 25 Hz")
    parser.add_argument("--min_duration", type=int, default=10,
                        help="最短段长，单位为原始 100 Hz 帧")
    parser.add_argument("--n_states", type=int, default=0,
                        help="期望模态数；0 = 按 gamma 阈值自动")
    parser.add_argument("--auto_sigma", type=float, default=2.0,
                        help="自动阈值 = mean(gamma) + auto_sigma * std(gamma)")
    parser.add_argument("--k", type=int, default=10, help="近邻数")
    parser.add_argument("--seed", type=int, default=0)
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

    if latent.ndim != 2:
        raise ValueError(f"latent 应为二维数组，实际 {latent.shape}")
    if not np.isfinite(latent).all():
        raise ValueError("latent 中包含 NaN 或 Inf")

    step = args.downsample
    latent = np.ascontiguousarray(latent[::step])
    time_ds = time_full[::step]

    scale = latent.std(axis=0)
    scale[scale == 0] = 1.0
    X = latent / scale

    T = len(X)
    min_len = max(2, int(round(args.min_duration / step)))
    print(f"开始 density-peaks 分割：T={T}（降采样 {step}），"
          f"min_duration={min_len} 帧，k={args.k}，n_states={args.n_states or 'auto'}")

    t0 = time.perf_counter()
    labels, peaks, rho, delta = density_peaks_cluster(
        X, n_states=args.n_states, k=args.k, auto_sigma=args.auto_sigma)
    print(f"  聚类完成：{len(np.unique(labels))} 个模态，"
          f"耗时 {time.perf_counter() - t0:.1f}s")

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
        out_path = str(p.parent / f"{rec}_dpc_states.npz")

    np.savez_compressed(
        out_path,
        states=labels.astype(np.int32),
        time=time_ds,
        latent=latent.astype(np.float32),
        centers=centers.astype(np.float32),
        segment_starts=starts.astype(np.int32),
        segment_lengths=lengths.astype(np.int32),
        rho=rho.astype(np.float32),
        delta=delta.astype(np.float32),
        downsample=step,
        min_duration=args.min_duration,
        n_states=len(np.unique(labels)),
        k=args.k,
        seed=args.seed,
    )

    print(f"结果已保存: {out_path}")
    print(f"  有效模态数: {len(np.unique(labels))}")
    print(f"  片段数: {len(lengths)}")
    print(f"  最短段长: {lengths.min()} 帧（降采样后）")
    print(f"  中位段长: {np.median(lengths):.1f} 帧（降采样后）")
    gamma = rho * delta
    top = np.argsort(-gamma)[:20]
    print("  gamma top-20: " + ", ".join(f"{gamma[i]:.3g}" for i in top))

if __name__ == "__main__":
    main()