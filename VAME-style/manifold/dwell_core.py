"""
Dwell-core segmentation: spatiotemporal persistence replaces the water level.

Rule (per frame i, standardized latent space):
  rho_i = #{j : ||x_j - x_i|| <= r}                (spatial neighbor count)
  tau_i = #{j : |t_j - t_i| <= W and ||x_j-x_i|| <= r}   (spatiotemporal count)

  rho_i < k_spatial          ->  sea    (-1)
  rho_i >= k_spatial,
  tau_i >= k_time            ->  dwell core (connected component id 1..K)
  otherwise                  ->  transit (0)

Dwell cores are connected components over dwell frames: edges link frames
less than W apart in time and within r in space (same definition as tau),
so one component = one sustained stay in one region of the manifold.

Outputs (next to the input hierarchy npz):
  <rec>_dwell.npz      klass (-1 sea / 0 transit / 1 dwell), dwell_id
                       (-1/0/1..K), time, latent, params
  <rec>_dwell_map.png      three-class map on the shared 2D embedding
  <rec>_dwell_timeline.png one timeline strip (sea/transit/dwell cores)

Usage:
  python dwell_core.py --hier_path ..._islands_hierarchy.npz
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
import scipy.sparse
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


def region_labels(dwell, X, r_link, min_frames):
    """Cluster dwell frames spatially (recurrent visits merge into one
    region). Edges = pairs within r_link in X. Regions >= min_frames get
    ids 1..K; smaller ones are demoted to transit (id 0)."""
    idx = np.nonzero(dwell)[0]
    if len(idx) == 0:
        return np.zeros(len(dwell), dtype=np.int32)
    sub = cKDTree(X[idx])
    pairs = sub.sparse_distance_matrix(sub, r_link).tocoo()

    parent = np.arange(len(idx))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in zip(pairs.row, pairs.col):
        if u < v:
            ru, rv = find(int(u)), find(int(v))
            if ru != rv:
                parent[rv] = ru

    roots = np.array([find(k) for k in range(len(idx))])
    _, inv, counts = np.unique(roots, return_inverse=True,
                               return_counts=True)
    big = counts >= min_frames
    label = np.zeros(len(counts), dtype=np.int32)
    # rank big regions by size: 1 = largest
    order = np.argsort(-counts)
    rank = np.zeros(len(counts), dtype=np.int32)
    for r_i, comp in enumerate([c for c in order if big[c]], start=1):
        rank[comp] = r_i
    dwell_id = np.zeros(len(dwell), dtype=np.int32)
    dwell_id[idx] = rank[inv]
    return dwell_id


def dwell_core(latent, W, r, k_spatial, k_time, min_frames,
               smooth_sigma=12.0, link_mult=2.5, knn=30):
    """Return (klass, dwell_id, stats). klass: -1 sea, 0 transit, 1 dwell.

    latent is smoothed along time first (Gaussian, sigma in frames) to remove
    per-frame jitter (median raw step ~1.2 sd >> behavioral drift scale),
    then standardized; r=None -> adaptive (1.5 x median dist-to-30NN).
    Land/sea: rho_i = #points within fixed radius r (spatial density axis).
    Persistence: tau_i = #{j in [i-W, i+W] : j in kNN(i)} -- kNN-graph
    proximity adapts to curved / anisotropic geometry (e.g. periodic orbits
    whose loop-to-loop chord distance far exceeds local spacing).
    dwell_id: 0 transit, 1..K spatial regions (ranked by size).
    """
    N, D = latent.shape
    if smooth_sigma and smooth_sigma > 0:
        latent_s = gaussian_filter1d(latent, smooth_sigma, axis=0,
                                     mode="nearest")
    else:
        latent_s = latent
    mu = latent_s.mean(axis=0)
    sd = latent_s.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    X = (latent_s - mu) / sd

    if r is None:
        tree0 = cKDTree(X)
        d30, _ = tree0.query(X, k=31)
        r = 1.5 * float(np.median(d30[:, -1]))
        print(f"adaptive radius r = {r:.3f} (1.5 x median dist-to-30NN, "
              f"smoothed)")

    tree = cKDTree(X)
    pairs = tree.sparse_distance_matrix(tree, r).tocoo()
    rho = np.bincount(np.concatenate([pairs.row, pairs.col]),
                      minlength=N)

    # tau via kNN graph (see docstring)
    _, idx_k = tree.query(X, k=knn + 1)
    S = scipy.sparse.csr_matrix(
        (np.ones(N * knn, dtype=np.int8),
         (np.repeat(np.arange(N), knn), idx_k[:, 1:].ravel())),
        shape=(N, N))
    tau = np.zeros(N, dtype=np.int32)
    for off in range(-W, W + 1):
        d = S.diagonal(off)
        if off >= 0:
            tau[:len(d)] += d
        else:
            tau[-off:-off + len(d)] += d

    sea = rho < k_spatial
    dwell = (~sea) & (tau >= k_time)
    transit = (~sea) & (~dwell)

    dwell_id = region_labels(dwell, X, r * link_mult, min_frames)
    # reclassify tiny regions back to transit
    dwell = dwell_id > 0
    transit = (~sea) & (~dwell)

    klass = np.full(N, -1, dtype=np.int32)
    klass[transit] = 0
    klass[dwell] = 1

    stats = dict(rho=rho, tau=tau, W=W, r=r, k_spatial=k_spatial,
                 k_time=k_time, min_frames=min_frames, knn=knn,
                 frac_sea=float(sea.mean()),
                 frac_transit=float(transit.mean()),
                 frac_dwell=float(dwell.mean()),
                 n_cores=int(dwell_id.max()))
    return klass, dwell_id, stats, X


# ----------------------------- visualization -------------------------------

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


def load_shared_embedding(latent_dir, recording, latent):
    p = Path(latent_dir) / f"{recording}_embedding.npy"
    if p.exists():
        return np.load(p)
    import umap
    print(f"computing shared embedding -> {p}")
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1,
                    random_state=42).fit_transform(latent)
    np.save(p, emb)
    return emb


def plot_map(emb, klass, dwell_id, out_path, alpha_mult=8.0,
             outline_min=150, max_outline=15, title_space="UMAP"):
    from matplotlib.collections import LineCollection
    from scipy.spatial import cKDTree, Delaunay

    sea = klass == -1
    transit = klass == 0
    dwell = klass == 1
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.scatter(emb[sea][:, 0], emb[sea][:, 1], s=1, c="0.85",
               alpha=0.35, linewidths=0, zorder=2)
    ax.scatter(emb[transit][:, 0], emb[transit][:, 1], s=3, c="#E69F00",
               alpha=0.45, linewidths=0, zorder=3)
    ax.scatter(emb[dwell][:, 0], emb[dwell][:, 1], s=5, c="#0072B2",
               alpha=0.85, linewidths=0, zorder=5)

    def alpha_shape_segments(pts, alpha_mult, max_pts=2000, seed=0):
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

    # outlines + labels only for sizable regions (ranked by size)
    sizes = np.bincount(dwell_id[dwell_id > 0])
    K = len(sizes)
    big = [u for u in np.argsort(-sizes) if sizes[u] >= outline_min]
    for rank, u in enumerate(big[:max_outline], start=1):
        mask = dwell_id == u
        color_u = cmap((rank - 1) / max(min(len(big), max_outline) - 1, 1))
        pts = emb[mask]
        segs = alpha_shape_segments(pts, alpha_mult)
        if segs is not None:
            ax.add_collection(LineCollection(segs, colors=[color_u],
                                             linewidths=1.8, zorder=5))
        cx, cy = pts.mean(axis=0)
        ax.plot(cx, cy, marker="x", color="black", markersize=10,
                markeredgewidth=2, zorder=6)
        ax.text(cx, cy, f" {rank}", fontsize=12, fontweight="bold",
                ha="left", va="center", zorder=6,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="black", pad=1.5))

    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="0.6",
                     markersize=8, label=f"sea ({int(sea.sum())} frames)"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#E69F00",
                     markersize=8, label=f"transit ({int(transit.sum())})"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
                     markersize=8,
                     label=f"dwell cores ({int((klass == 1).sum())}, {K} regions)")]
    ax.legend(handles=legend, loc="lower right", fontsize=10,
              framealpha=0.9)
    ax.set_title(f"Dwell cores / transit / sea ({title_space} 2D view)",
                 fontsize=13)
    ax.set_xlabel("embedding dim 1")
    ax.set_ylabel("embedding dim 2")
    ax.set_aspect("equal")
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


def plot_timeline(time_ds, klass, dwell_id, out_path):
    K = int(dwell_id.max())
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(15, 1.8))
    starts = [0]
    for t in range(1, len(klass)):
        if klass[t] != klass[t - 1] or dwell_id[t] != dwell_id[t - 1]:
            starts.append(t)
    starts.append(len(klass))
    for a, b in zip(starts[:-1], starts[1:]):
        c = klass[a]
        if c == -1:
            color = "0.9"
        elif c == 0:
            color = "#E69F00"
        else:
            color = "#0072B2"
        ax.hlines(0, time_ds[a], time_ds[b - 1], colors=color, linewidth=5)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"dwell-core timeline ({K} cores)", fontsize=12)
    ax.spines[["left", "right", "top"]].set_visible(False)
    fig.tight_layout()
    save_fig(fig, out_path)
    plt.close(fig)


# ---------------------------------- main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Dwell-core segmentation")
    ap.add_argument("--hier_path", required=True,
                    help="*_islands_hierarchy.npz (for latent/time)")
    ap.add_argument("--window", type=int, default=7,
                    help="temporal half-window W (frames; 7 ~= 0.3 s at 25 Hz)")
    ap.add_argument("--radius", type=float, default=None,
                    help="spatial radius r in standardized units "
                         "(default: 1.5 x median dist-to-30NN)")
    ap.add_argument("--k_spatial", type=int, default=30,
                    help="min spatial neighbors to be 'land' (else sea)")
    ap.add_argument("--k_time", type=int, default=5,
                    help="min spatiotemporal neighbors to be dwell core")
    ap.add_argument("--min_frames", type=int, default=10,
                    help="min frames for a dwell component to keep its id")
    ap.add_argument("--smooth_sigma", type=float, default=12.0,
                    help="temporal Gaussian smoothing sigma (frames; "
                         "12 ~= 0.5 s at 25 Hz), removes per-frame jitter")
    ap.add_argument("--link_mult", type=float, default=2.5,
                    help="dwell-region linking radius = link_mult x r "
                         "(smaller -> more regions)")
    ap.add_argument("--knn", type=int, default=30,
                    help="k for kNN-graph persistence (tau counts window "
                         "frames inside your k spatial nearest neighbors)")
    args = ap.parse_args()

    hier = Path(args.hier_path)
    d = np.load(hier)
    latent = d["latent"].astype(np.float64)
    time_ds = d["time"] if "time" in d else np.arange(len(latent))

    recording = hier.name
    for suffix in ("_islands_hierarchy.npz", "_ws_hierarchy.npz",
                   "_hierarchy.npz"):
        if recording.endswith(suffix):
            recording = recording[:-len(suffix)]
            break
    out_dir = hier.parent

    # radius: None -> adaptive inside dwell_core (on smoothed, standardized
    # latent, 1.5 x median dist-to-30NN)
    t0 = time.time()
    klass, dwell_id, stats, X = dwell_core(
        latent, args.window, args.radius, args.k_spatial, args.k_time,
        args.min_frames, smooth_sigma=args.smooth_sigma,
        link_mult=args.link_mult, knn=args.knn)
    print(f"classify done in {time.time() - t0:.1f}s")
    N = len(klass)
    print(f"frames      = {N}")
    print(f"sea         = {(klass == -1).sum():6d}  "
          f"({stats['frac_sea'] * 100:.1f}%)")
    print(f"transit     = {(klass == 0).sum():6d}  "
          f"({stats['frac_transit'] * 100:.1f}%)")
    print(f"dwell cores = {(klass == 1).sum():6d}  "
          f"({stats['frac_dwell'] * 100:.1f}%)  "
          f"in {stats['n_cores']} regions")
    sizes = np.bincount(dwell_id[dwell_id > 0])
    print("core sizes (frames):",
          np.sort(sizes)[::-1][:20].tolist(), "...")

    out_npz = out_dir / f"{recording}_dwell.npz"
    np.savez_compressed(
        out_npz,
        klass=klass.astype(np.int32), dwell_id=dwell_id,
        time=time_ds, latent=latent.astype(np.float32),
        W=args.window, r=stats["r"], k_spatial=args.k_spatial,
        k_time=args.k_time, min_frames=args.min_frames,
        smooth_sigma=args.smooth_sigma, knn=args.knn)
    print(f"Saved: {out_npz}")

    emb = load_shared_embedding(out_dir, recording, latent)
    plot_map(emb, klass, dwell_id, str(out_dir / f"{recording}_dwell_map.png"),
             title_space="UMAP")

    # faithful-geometry view: PCA-2D of the SAME smoothed, standardized
    # latent that the classifier used (linear projection, no tearing)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca2 = U[:, :2] * S[:2]
    var2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    print(f"PCA-2D variance explained: {var2 * 100:.1f}%")
    plot_map(pca2, klass, dwell_id,
             str(out_dir / f"{recording}_dwell_map_pca.png"),
             title_space="PCA")

    plot_timeline(time_ds, klass, dwell_id,
                  str(out_dir / f"{recording}_dwell_timeline.png"))


if __name__ == "__main__":
    main()
