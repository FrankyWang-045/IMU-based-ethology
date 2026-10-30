"""
Zoomed vortex visualization around a density peak.

For the recording's shared UMAP embedding, picks a density peak (default:
global argmax of the smoothed density grid; override with --peak "x y"),
opens a square zoom window around it and produces ONE figure with three
panels (saved next to the latent npz as <rec>_vortex_zoom.png):

  A. density contours + extrapolated drift streamlines  b(x), tau -> 0
  B. real trajectory segments passing through the window
     (time-colored lines + direction arrows): if the peak is a limit-cycle
     attractor, segments should visibly loop around it
  C. vorticity curl(b) field (red = CCW, blue = CW) + streamlines

Usage:
  python plot_vortex.py --latent_path <...>_latent.npz [--peak "x y"]
                        [--window 3.0] [--downsample 4]
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
    """Returns (D[y,x], extent). histogram2d axes are [x,y]; transpose so
    row axis = y, matching hand-built arrays used by contour/streamplot."""
    mn, mx = emb.min(axis=0), emb.max(axis=0)
    pad = 0.05 * (mx - mn)
    H, xedges, yedges = np.histogram2d(
        emb[:, 0], emb[:, 1], bins=res,
        range=[[mn[0] - pad[0], mx[0] + pad[0]],
               [mn[1] - pad[1], mx[1] + pad[1]]])
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


def extrapolate_drift(fields, taus, small=(0.12, 0.25, 0.5)):
    """Through-origin least squares over small taus: b = sum(tau*m)/sum(tau^2)."""
    idx = [i for i, t in enumerate(taus)
           if any(abs(t - s) < 1e-9 for s in small)]
    num = sum(taus[i] * fields[i] for i in idx)
    den = sum(taus[i] ** 2 for i in idx)
    return num / den


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


# ------------------------------------------------------- trajectory runs
def window_runs(emb, win, min_len=8):
    """Consecutive index runs with emb inside the window [x0,x1,y0,y1]."""
    x0, x1, y0, y1 = win
    inside = (emb[:, 0] >= x0) & (emb[:, 0] <= x1) & \
             (emb[:, 1] >= y0) & (emb[:, 1] <= y1)
    d = np.diff(inside.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if inside[0]:
        starts = np.r_[0, starts]
    if inside[-1]:
        ends = np.r_[ends, len(inside)]
    runs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]
    runs.sort(key=lambda se: -(se[1] - se[0]))
    return runs


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Zoomed vortex around a density peak")
    ap.add_argument("--latent_path", required=True)
    ap.add_argument("--peak", type=float, nargs=2, default=None,
                    help="peak position in embedding coords (default: density argmax)")
    ap.add_argument("--window", type=float, default=3.0,
                    help="zoom half-width in embedding units (default 3)")
    ap.add_argument("--max_runs", type=int, default=25,
                    help="max trajectory runs drawn in panel B (longest first)")
    ap.add_argument("--min_dwell", type=int, default=25,
                    help="min run length in frames (~min dwell time in window)")
    ap.add_argument("--downsample", type=int, default=4)
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

    # ---- pick the peak ----
    xs_d = np.linspace(extent[0], extent[1], D2d.shape[1])
    ys_d = np.linspace(extent[2], extent[3], D2d.shape[0])
    if args.peak is None:
        iy, ix = np.unravel_index(np.argmax(D2d), D2d.shape)
        px, py = float(xs_d[ix]), float(ys_d[iy])
    else:
        px, py = args.peak
        ix = int(np.clip(np.searchsorted(xs_d, px), 0, len(xs_d) - 1))
        iy = int(np.clip(np.searchsorted(ys_d, py), 0, len(ys_d) - 1))
    w = args.window
    win = (px - w, px + w, py - w, py + w)
    print(f"peak = ({px:.2f}, {py:.2f}), window half-width = {w}")

    # ---- drift field b(x) ----
    ladder = [0.12, 0.25, 0.5, 1.0, 2.0]
    xc, yc, fields, cnt0 = drift_ladder(emb, dt, ladder)
    b = extrapolate_drift(fields, ladder)
    b_f = gap_fill(b)
    bn = np.linalg.norm(b, axis=-1)

    U, V = b_f[..., 0], b_f[..., 1]
    div = np.gradient(U, xc, axis=1) + np.gradient(V, yc, axis=0)
    curl = np.gradient(V, xc, axis=1) - np.gradient(U, yc, axis=0)
    valid = ~np.isnan(bn) & (cnt0 >= 10)

    # crop zoom slices
    ix0 = int(np.clip(np.searchsorted(xc, win[0]) - 1, 0, len(xc) - 1))
    ix1 = int(np.clip(np.searchsorted(xc, win[1]) + 1, 1, len(xc)))
    iy0 = int(np.clip(np.searchsorted(yc, win[2]) - 1, 0, len(yc) - 1))
    iy1 = int(np.clip(np.searchsorted(yc, win[3]) + 1, 1, len(yc)))
    Xc, Yc = np.meshgrid(xc[ix0:ix1], yc[iy0:iy1])
    Uz = np.ma.masked_where(~valid[iy0:iy1, ix0:ix1], U[iy0:iy1, ix0:ix1])
    Vz = np.ma.masked_where(~valid[iy0:iy1, ix0:ix1], V[iy0:iy1, ix0:ix1])
    div_z = div[iy0:iy1, ix0:ix1]
    curl_z = curl[iy0:iy1, ix0:ix1]

    # density contours on the same zoom window
    id0 = int(np.clip(np.searchsorted(xs_d, win[0]) - 1, 0, len(xs_d) - 1))
    id1 = int(np.clip(np.searchsorted(xs_d, win[1]) + 1, 1, len(xs_d)))
    jd0 = int(np.clip(np.searchsorted(ys_d, win[2]) - 1, 0, len(ys_d) - 1))
    jd1 = int(np.clip(np.searchsorted(ys_d, win[3]) + 1, 1, len(ys_d)))
    Dz = D2d[jd0:jd1, id0:id1]
    ext_z = [xs_d[id0], xs_d[id1 - 1], ys_d[jd0], ys_d[jd1 - 1]]

    # peak-region diagnostics
    div_pk = float(np.nanmean(div_z[np.isfinite(div_z)]))
    print(f"mean divergence in window = {div_pk:.3f} /s "
          f"(negative = net inflow)")

    # ---- panel B: real trajectory runs ----
    runs = window_runs(emb, win, min_len=args.min_dwell)
    print(f"{len(runs)} trajectory runs (>={args.min_dwell} frames "
          f"~ >={args.min_dwell*dt:.1f}s dwell) pass through window; "
          f"drawing up to {args.max_runs} longest")

    # ------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # A: density + streamlines
    ax = axes[0]
    cf = ax.contourf(Dz, extent=ext_z, levels=12, cmap="Greys", alpha=0.55)
    ax.contour(Dz, extent=ext_z, levels=6, colors="k", linewidths=0.4,
               alpha=0.5)
    ax.streamplot(Xc, Yc, Uz, Vz, color="tab:red", density=2.2,
                  linewidth=1.0, arrowsize=1.2)
    ax.plot(px, py, marker="*", ms=18, mfc="gold", mec="k")
    ax.set_title("A. drift b(x) streamlines over density\n"
                 "(tau->0 extrapolation; masked where data sparse)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # B: real trajectories, time-colored
    ax = axes[1]
    ax.contour(Dz, extent=ext_z, levels=8, colors="k", linewidths=0.4,
               alpha=0.45)
    drawn = 0
    t0 = time_ds[0]
    tspan = max(time_ds[-1] - time_ds[0], 1e-9)
    for (s, e) in runs[:args.max_runs]:
        pts = emb[s:e]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)          # (L,2,2)
        tc = (time_ds[s:e - 1] - t0) / tspan
        lc = LineCollection(segs, array=tc, cmap="viridis",
                            linewidths=1.0, alpha=0.85)
        ax.add_collection(lc)
        # direction arrows at 25/50/75% along the run
        for m in {len(pts) // 4, len(pts) // 2, 3 * len(pts) // 4}:
            if m >= 1:
                ax.annotate("", xy=pts[m], xytext=pts[m - 1],
                            arrowprops=dict(arrowstyle="-|>", color="0.2",
                                            lw=0.8, mutation_scale=8))
        drawn += 1
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(t0, t0 + tspan))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="time (s)")
    ax.plot(px, py, marker="*", ms=18, mfc="gold", mec="k")
    ax.set_title(f"B. real trajectory segments through window\n"
                 f"({drawn} runs; arrows = direction; color = time)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # C: vorticity
    ax = axes[2]
    lim = np.nanpercentile(np.abs(curl_z[np.isfinite(curl_z)]), 95)
    lim = max(lim, 1e-6)
    cf = ax.contourf(Xc, Yc, np.clip(curl_z, -lim, lim), levels=20,
                     cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(cf, ax=ax, label="curl b  (1/s)")
    ax.streamplot(Xc, Yc, Uz, Vz, color="k", density=1.6,
                  linewidth=0.7, arrowsize=1.0)
    ax.plot(px, py, marker="*", ms=18, mfc="gold", mec="k")
    ax.set_title("C. vorticity curl(b)\n(red = CCW, blue = CW)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    for ax in axes:
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_aspect("equal")

    fig.suptitle(f"{recording} | zoom around peak ({px:.2f}, {py:.2f}) | "
                 f"window +/-{w}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, str(latent_dir / f"{recording}_vortex_zoom.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
