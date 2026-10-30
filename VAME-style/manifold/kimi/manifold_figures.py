"""
Latent-manifold visualization for one recording.

Produces three figures (saved next to the latent npz):
  1. <rec>_terrain.png        2D density terrain + point cloud
  2. <rec>_drift.png          Lag-tau drift field (streamlines, speed-colored)
  3. <rec>_path_density.png   Trajectory throughput raster (log scale)

The display embedding is the shared coordinate system used across the
project: UMAP(n_neighbors=15, min_dist=0.1, random_state=42) on the
[::step] analysis frame set. It is computed once and cached as
<rec>_embedding.npy so all scripts/recordings reuse the same convention.

Usage:
  python manifold_figures.py --latent_path <...>_latent.npz [--downsample 4]
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------- utils
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
    """Shared-coordinate embedding (same params as visualize_umap_terrain)."""
    p = Path(latent_dir) / f"{recording}_embedding.npy"
    if p.exists():
        print(f"using cached embedding: {p}")
        return np.load(p)
    import umap
    print("computing shared embedding UMAP(15, 0.1, rs42) ...")
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


# ---------------------------------------------------------------- drift
def _bin_means(emb, disp, edges, n_bins):
    """Mean displacement per spatial bin; layout [y, x, (dx, dy)]."""
    ix = np.clip(np.digitize(emb[:, 0], edges[0]) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(emb[:, 1], edges[1]) - 1, 0, n_bins - 1)
    flat = iy * n_bins + ix
    cnt = np.bincount(flat, minlength=n_bins * n_bins).astype(float)
    mx = np.bincount(flat, weights=disp[:, 0],
                     minlength=n_bins * n_bins) / np.maximum(cnt, 1)
    my = np.bincount(flat, weights=disp[:, 1],
                     minlength=n_bins * n_bins) / np.maximum(cnt, 1)
    M = np.stack([mx, my], axis=-1).reshape(n_bins, n_bins, 2)
    M[cnt.reshape(n_bins, n_bins) < 1] = np.nan
    return M, cnt.reshape(n_bins, n_bins)


def drift_ladder(emb, dt, taus, n_bins=50, min_count=10):
    """Per-bin mean lag-tau displacement for each tau in `taus`.

    Returns (xc, yc, fields, cnt0): cnt0 is the per-bin sample count of the
    first (smallest-k, largest-sample) tau, for weighting diagnostics.
    """
    mn, mx = emb.min(axis=0), emb.max(axis=0)
    pad = 0.02 * (mx - mn)
    edges = [np.linspace(mn[0] - pad[0], mx[0] + pad[0], n_bins + 1),
             np.linspace(mn[1] - pad[1], mx[1] + pad[1], n_bins + 1)]
    xc = 0.5 * (edges[0][:-1] + edges[0][1:])
    yc = 0.5 * (edges[1][:-1] + edges[1][1:])
    fields = []
    cnt0 = None
    for tau in taus:
        k = max(1, int(round(tau / dt)))
        M, cnt = _bin_means(emb[:-k], emb[k:] - emb[:-k], edges, n_bins)
        M[cnt < min_count] = np.nan
        fields.append(M)
        if cnt0 is None:
            cnt0 = cnt
    return xc, yc, fields, cnt0


def gap_fill(F, max_dist=3):
    """Nearest-fill nan holes <= max_dist cells (old-scipy compatible)."""
    from scipy.ndimage import distance_transform_edt
    bad = np.isnan(F[..., 0])
    if not bad.any() or bad.all():
        return np.nan_to_num(F, nan=0.0)
    dist = distance_transform_edt(bad)
    out = distance_transform_edt(bad, return_indices=True)
    ind = out[-1] if isinstance(out, tuple) else out
    near = dist <= max_dist
    F = F.copy()
    F[near] = F[tuple(ind)][near]
    return np.nan_to_num(F, nan=0.0)


def extrapolate_drift(fields, taus, small=(0.12, 0.25, 0.5)):
    """Through-origin least squares over small taus: b = sum(tau*m)/sum(tau^2).

    Large-tau conditional means are biased toward the global center of
    mass (v_tau -> E[x] - x), so only the small-tau range is used.
    """
    idx = [i for i, t in enumerate(taus)
           if any(abs(t - s) < 1e-9 for s in small)]
    num = sum(taus[i] * fields[i] for i in idx)
    den = sum(taus[i] ** 2 for i in idx)
    return num / den


# ------------------------------------------------------- path throughput
def path_density(emb, extent, res=500, samples_per_seg=16):
    """Rasterize every consecutive-frame segment; brightness = throughput."""
    acc = np.zeros((res, res), dtype=np.float64)
    a = emb[:-1]
    b = emb[1:]
    ts = np.linspace(0.0, 1.0, samples_per_seg)[None, :, None]
    pts = a[:, None, :] * (1 - ts) + b[:, None, :] * ts     # (N, S, 2)
    pts = pts.reshape(-1, 2)
    gx = (pts[:, 0] - extent[0]) / (extent[1] - extent[0]) * res
    gy = (pts[:, 1] - extent[2]) / (extent[3] - extent[2]) * res
    ok = (gx >= 0) & (gx < res) & (gy >= 0) & (gy < res)
    np.add.at(acc, (np.floor(gy[ok]).astype(int),
                    np.floor(gx[ok]).astype(int)), 1.0)
    acc = gaussian_filter(acc, sigma=1.0)
    return acc


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Latent-manifold figures")
    ap.add_argument("--latent_path", required=True)
    ap.add_argument("--downsample", type=int, default=4,
                    help="analysis-frame decimation (default 4, 100->25 Hz)")
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.5, 2.0], help="drift lag in seconds")
    args = ap.parse_args()

    latent_path = Path(args.latent_path)
    latent_dir = latent_path.parent
    recording = latent_path.name.replace("_latent.npz", "")

    d = np.load(latent_path)
    latent = d["latent"].astype(np.float64)
    time_full = d["time"] if "time" in d else np.arange(len(latent))
    step = args.downsample
    latent_ds = np.ascontiguousarray(latent[::step])
    time_ds = np.ascontiguousarray(time_full[::step])
    dt = float(np.median(np.diff(time_ds)))
    print(f"recording={recording}  N={len(latent_ds)}  dt={dt*1e3:.1f} ms")

    emb = load_or_make_embedding(latent_dir, recording, latent_ds)
    D2d, extent = density_grid(emb)
    sel = np.arange(len(emb))
    if len(sel) > 40000:
        sel = np.random.default_rng(0).choice(len(emb), 40000, replace=False)

    # ---------------- figure 1: terrain ----------------
    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(D2d, extent=extent, levels=20, cmap="terrain", alpha=0.9)
    fig.colorbar(cf, ax=ax, label="density")
    ax.scatter(emb[sel, 0], emb[sel, 1], s=1, c="black", alpha=0.25,
               linewidths=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{recording} | latent density terrain")
    save_fig(fig, str(latent_dir / f"{recording}_terrain.png"))
    plt.close(fig)

    # ---------------- figure 2: drift field ----------------
    # tau ladder -> extrapolate small-tau drift b(x); diagnostic: apparent
    # convergence point migrates toward the global mean as tau grows
    ladder = [0.12, 0.25, 0.5, 1.0, 2.0]
    xc, yc, fields, cnt0 = drift_ladder(emb, dt, ladder)
    b = extrapolate_drift(fields, ladder)              # [y,x,2], units/s
    b_f = gap_fill(b)
    bn = np.linalg.norm(b, axis=-1)                    # raw speed (nan ok)
    spd_disp = np.ma.masked_invalid(bn)

    # divergence / curl of b: convergence-to-peaks is a statement about
    # NET flow (divergence), not pointwise gradient alignment
    U_b, V_b = b_f[..., 0], b_f[..., 1]
    div = np.gradient(U_b, xc, axis=1) + np.gradient(V_b, yc, axis=0)
    curl = np.gradient(V_b, xc, axis=1) - np.gradient(U_b, yc, axis=0)
    valid = ~np.isnan(bn) & (cnt0 >= 10)

    # theory check: b should align with grad log density  (b ~ (D/2) grad ln r)
    logD = np.log(D2d + 1e-6)
    xs_d = np.linspace(extent[0], extent[1], D2d.shape[1])
    ys_d = np.linspace(extent[2], extent[3], D2d.shape[0])
    gy, gx = np.gradient(logD, ys_d, xs_d)
    col = np.clip(np.searchsorted(xs_d, xc), 1, len(xs_d) - 1) - 1
    row = np.clip(np.searchsorted(ys_d, yc), 1, len(ys_d) - 1) - 1
    GX, GY = gx[np.ix_(row, col)], gy[np.ix_(row, col)]
    gn = np.sqrt(GX ** 2 + GY ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_bg = (b[..., 0] * GX + b[..., 1] * GY) / (bn * gn)
    good = (bn > np.nanpercentile(bn, 50)) & (gn > 1e-3) & ~np.isnan(cos_bg)
    mean_cos = float(np.nanmean(np.where(good, cos_bg, np.nan)))
    # where does the extrapolated drift slow down (attractor candidates)?
    dens_bin = D2d[np.ix_(row, col)]
    pk = np.unravel_index(np.argmax(dens_bin), dens_bin.shape)
    bmin = np.unravel_index(np.nanargmin(bn), bn.shape)
    dist_peak_bmin = float(np.hypot(xc[bmin[1]] - xc[pk[1]],
                                    yc[bmin[0]] - yc[pk[0]]))
    # net inflow on dense sets vs sparse sets (the user's prediction:
    # dense = high-frequency behavior = attractor => divergence < 0)
    d_valid = dens_bin[valid]
    thr_hi = np.percentile(d_valid, 80)
    thr_lo = np.percentile(d_valid, 20)
    div_top = float(np.mean(div[valid & (dens_bin >= thr_hi)]))
    div_bot = float(np.mean(div[valid & (dens_bin <= thr_lo)]))
    curl_top = float(np.mean(np.abs(curl)[valid & (dens_bin >= thr_hi)]))
    curl_bot = float(np.mean(np.abs(curl)[valid & (dens_bin <= thr_lo)]))
    mu = emb.mean(axis=0)
    print(f"[drift] mean cos(b, grad ln rho) = {mean_cos:.3f}")
    print(f"[drift] divergence: top-density bins {div_top:.3f} /s, "
          f"bottom-density bins {div_bot:.3f} /s "
          f"(negative = net inflow)")
    print(f"[drift] |curl|: top {curl_top:.3f}, bottom {curl_bot:.3f} /s "
          f"(high on ridges = circulation)")
    print(f"[drift] density peak bin @ ({xc[pk[1]]:.2f}, {yc[pk[0]]:.2f}); "
          f"slowest-drift bin @ ({xc[bmin[1]]:.2f}, {yc[bmin[0]]:.2f}); "
          f"distance = {dist_peak_bmin:.2f} UMAP units")
    print(f"[drift] global mean = ({mu[0]:.2f}, {mu[1]:.2f})")

    fig, axes = plt.subplots(1, 3, figsize=(27, 8))
    # left: raw small-tau velocity (what a single-tau plot would show)
    ax = axes[0]
    raw_tau = args.taus[0] if args.taus else 0.5
    raw_i = min(range(len(ladder)), key=lambda i: abs(ladder[i] - raw_tau))
    v_raw = fields[raw_i] / ladder[raw_i]
    ax.contourf(D2d, extent=extent, levels=12, cmap="Greys", alpha=0.25,
                vmin=0, vmax=1)
    ax.contourf(xc, yc, np.ma.masked_invalid(np.linalg.norm(v_raw, axis=-1)),
                levels=15, cmap="viridis", alpha=0.85)
    vr = gap_fill(v_raw)
    ax.streamplot(xc, yc, vr[..., 0], vr[..., 1], color="white",
                  density=1.6, linewidth=1.0, arrowsize=0.9, zorder=5)
    ax.set_title(f"raw single-tau field (tau={ladder[raw_i]:g}s)\n"
                 "(biased: large-tau limit points to global mean)")
    # middle: extrapolated drift
    ax = axes[1]
    ax.contourf(D2d, extent=extent, levels=12, cmap="Greys", alpha=0.25,
                vmin=0, vmax=1)
    cf = ax.contourf(xc, yc, spd_disp, levels=15, cmap="viridis", alpha=0.85)
    fig.colorbar(cf, ax=ax, label="|drift b(x)| (UMAP units / s)")
    ax.streamplot(xc, yc, b_f[..., 0], b_f[..., 1], color="white",
                  density=1.6, linewidth=1.0, arrowsize=0.9, zorder=5)
    ax.plot(xc[pk[1]], yc[pk[0]], "w*", markersize=18, mew=2, zorder=6,
            label="density peak")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title(f"extrapolated drift b(x), tau->0\n"
                 f"mean cos(b, grad ln rho) = {mean_cos:.2f}; "
                 f"div: dense {div_top:.1f} vs sparse {div_bot:.1f} /s")
    # right: convergence-point migration diagnostic
    ax = axes[2]
    ax.contourf(D2d, extent=extent, levels=12, cmap="Greys", alpha=0.3,
                vmin=0, vmax=1)
    ax.scatter(emb[sel, 0], emb[sel, 1], s=1, c="black", alpha=0.15,
               linewidths=0)
    for i, tau in enumerate(ladder):
        vmag = np.where(np.isnan(fields[i][..., 0]), np.nan,
                        np.linalg.norm(fields[i], axis=-1) / tau)
        v_valid = ~np.isnan(vmag) & (cnt0 >= 10)
        if v_valid.sum() < 5:
            continue
        thr = np.nanpercentile(vmag[v_valid], 10)
        selm = v_valid & (vmag <= thr)
        w = cnt0[selm]
        cx = float(np.sum(xc[np.where(selm)[1]] * w) / np.sum(w))
        cy = float(np.sum(yc[np.where(selm)[0]] * w) / np.sum(w))
        ax.plot(cx, cy, "o", markersize=10,
                color=plt.cm.plasma(i / (len(ladder) - 1)), mec="black",
                zorder=6)
        ax.annotate(f"{tau:g}s", (cx, cy), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.plot(mu[0], mu[1], "*", color="red", markersize=20, mew=2, zorder=6)
    ax.annotate("global mean", (mu[0], mu[1]), textcoords="offset points",
                xytext=(8, -14), fontsize=10, color="red")
    ax.set_title("apparent convergence point vs tau\n"
                 "(migration toward global mean = estimator artifact)")
    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle(f"{recording} | drift-field estimator: raw vs tau-extrapolated")
    save_fig(fig, str(latent_dir / f"{recording}_drift.png"))
    plt.close(fig)

    # ---------------- figure 3: path throughput ----------------
    acc = path_density(emb, extent)
    acc_n = acc / max(acc.max(), 1e-12)
    # occupancy at the same grid & smoothing scale, for comparison
    occ, _, _ = np.histogram2d(
        emb[:, 0], emb[:, 1], bins=500,
        range=[[extent[0], extent[1]], [extent[2], extent[3]]])
    occ_s = gaussian_filter(occ, sigma=1.0)
    occ_n = occ_s / max(occ_s.max(), 1e-12)
    # excess traffic: throughput beyond what dwell time explains
    # (high on highways & loops, low inside attractor cores)
    excess = np.clip(acc_n - occ_n, 0.0, None)
    excess /= max(excess.max(), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, field, cmap_, ttl, cbar_lbl in [
        (axes[0], acc_n, "inferno", "raw throughput", "norm. segment count"),
        (axes[1], excess, "magma", "excess throughput "
         "(throughput - occupancy)", "excess (norm.)"),
    ]:
        cf = ax.contourf(field, extent=extent,
                         levels=np.linspace(0, 1, 21), cmap=cmap_)
        fig.colorbar(cf, ax=ax, label=cbar_lbl)
        ax.contour(D2d, extent=extent, levels=8, colors="cyan",
                   linewidths=0.3, alpha=0.5)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(ttl)
    fig.suptitle(f"{recording} | trajectory throughput "
                 f"(all {len(emb)-1} segments)")
    save_fig(fig, str(latent_dir / f"{recording}_path_density.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
