"""
Vortex / drift-field analysis in 3D PCA space, displayed in 2D UMAP coords.

Motivation: the 2D UMAP display space folds the manifold, so the drift
b(x) estimated there is distorted (patchy vorticity, fake convergence).
Here b(x) is estimated in the 3D PCA space of the latent (metric-preserving
orthogonal projection, same convention as watershed.py: unstandardized SVD,
fixed 3 components), then streamlines are integrated in 3D and projected
back to the shared UMAP display coordinates for comparison with the
2D-estimated version (plot_vortex.py).

Mapping between display (UMAP) coords and PCA coords is local kNN averaging
in both directions; no global inverse of UMAP is assumed.

Produces <rec>_vortex_zoom_pca3d.png in the latent dir:
  A. 2D-estimated b(x) streamlines (reference, same as plot_vortex.py)
  B. 3D-estimated b(x): integrated in PCA, projected to UMAP display
  C. 3D divergence of b, sampled over the zoom window (negative = inflow)

Also prints the decisive statistics in 3D (dense-vs-sparse divergence,
cos(b, grad ln rho)) for comparison with the 2D numbers.

Usage:
  python plot_vortex_pca3d.py --latent_path <...>_latent.npz \
      [--peak "x y"] [--window 2.5] [--downsample 4] [--bins 40]
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
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


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


def pca3(X):
    """Unstandardized SVD projection, 3 components (watershed.py convention)."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    cvar = np.cumsum(var) / var.sum()
    print(f"PCA: first 3 dims explain {cvar[2]*100:.1f}% variance")
    return U[:, :3] * S[:3]


def density_grid(emb, res=200, sigma=2):
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


# ------------------------------------------------------- local mappings
class LocalMap:
    """Local kNN-averaged map between UMAP display coords and PCA coords."""

    def __init__(self, emb, Z, k=10):
        self.emb, self.Z, self.k = emb, Z, k
        self.tree_emb = cKDTree(emb)
        self.tree_Z = cKDTree(Z)
        # typical local scales
        d, _ = self.tree_emb.query(emb, k=k + 1)
        self.h_emb = float(np.median(d[:, -1]))
        d, _ = self.tree_Z.query(Z, k=k + 1)
        self.h_Z = float(np.median(d[:, -1]))
        print(f"local scales: h_umap={self.h_emb:.3f}  h_pca={self.h_Z:.3f}")

    def emb_to_pca(self, pts, k=10):
        d, idx = self.tree_emb.query(np.atleast_2d(pts), k=k)
        w = np.exp(-(d / max(self.h_emb, 1e-9)) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        return (w[..., None] * self.Z[idx]).sum(axis=1)

    def pca_to_emb(self, pts, k=15):
        d, idx = self.tree_Z.query(np.atleast_2d(pts), k=k)
        h = d[:, -1:] + 1e-9
        w = np.exp(-(d / h) ** 2)
        w /= w.sum(axis=1, keepdims=True)
        return (w[..., None] * self.emb[idx]).sum(axis=1)


# ------------------------------------------------------------- 3D drift
def bin_means3(Z, disp, edges, n_bins):
    """Mean displacement per 3D bin. Returns M[y,x,z,3] wait -> [b0,b1,b2]
    laid out as (n_bins, n_bins, n_bins, 3) over (z0, z1, z2) axes."""
    ix = [np.clip(np.digitize(Z[:, j], edges[j]) - 1, 0, n_bins - 1)
          for j in range(3)]
    flat = (ix[0] * n_bins + ix[1]) * n_bins + ix[2]
    cnt = np.bincount(flat, minlength=n_bins ** 3).astype(float)
    Ms = []
    for j in range(3):
        m = np.bincount(flat, weights=disp[:, j],
                        minlength=n_bins ** 3) / np.maximum(cnt, 1)
        Ms.append(m.reshape(n_bins, n_bins, n_bins))
    return np.stack(Ms, axis=-1), cnt.reshape(n_bins, n_bins, n_bins)


def drift_ladder3(Z, dt, taus, n_bins=40, min_count=10):
    mn, mx = Z.min(axis=0), Z.max(axis=0)
    pad = 0.02 * (mx - mn)
    edges = [np.linspace(mn[j] - pad[j], mx[j] + pad[j], n_bins + 1)
             for j in range(3)]
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    fields, cnt0 = [], None
    for tau in taus:
        k = max(1, int(round(tau / dt)))
        M, cnt = bin_means3(Z[:-k], Z[k:] - Z[:-k], edges, n_bins)
        M[cnt < min_count] = np.nan
        fields.append(M)
        if cnt0 is None:
            cnt0 = cnt
    return centers, edges, fields, cnt0


def extrapolate_drift3(fields, taus, small=(0.12, 0.25, 0.5)):
    idx = [i for i, t in enumerate(taus)
           if any(abs(t - s) < 1e-9 for s in small)]
    num = sum(taus[i] * fields[i] for i in idx)
    den = sum(taus[i] ** 2 for i in idx)
    return num / den


def gap_fill3(F, cnt, max_dist=2):
    """Nearest-fill nan holes close to data (3D, old-scipy compatible)."""
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
def window_runs(emb, win, min_len=25):
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
    ap = argparse.ArgumentParser(
        description="3D-PCA drift field, projected to UMAP display")
    ap.add_argument("--latent_path", required=True)
    ap.add_argument("--peak", type=float, nargs=2, default=None)
    ap.add_argument("--window", type=float, default=2.5)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--max_runs", type=int, default=25)
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
    Z = pca3(latent_ds)
    lmap = LocalMap(emb, Z)

    D2d, extent = density_grid(emb)
    xs_d = np.linspace(extent[0], extent[1], D2d.shape[1])
    ys_d = np.linspace(extent[2], extent[3], D2d.shape[0])
    if args.peak is None:
        iy, ix = np.unravel_index(np.argmax(D2d), D2d.shape)
        px, py = float(xs_d[ix]), float(ys_d[iy])
    else:
        px, py = args.peak
    w = args.window
    win = (px - w, px + w, py - w, py + w)
    print(f"peak = ({px:.2f}, {py:.2f}), window half-width = {w}")

    # ================= 3D drift field =================
    ladder = [0.12, 0.25, 0.5, 1.0, 2.0]
    centers, edges, fields, cnt0 = drift_ladder3(
        Z, dt, ladder, n_bins=args.bins)
    b3 = extrapolate_drift3(fields, ladder)          # (B,B,B,3), units/s
    b3_f = gap_fill3(b3, cnt0)
    valid3 = cnt0 >= 10

    # interpolators (linear for field, nearest for validity)
    b3_interp = RegularGridInterpolator(
        centers, b3_f, bounds_error=False, fill_value=np.nan)
    valid3_interp = RegularGridInterpolator(
        centers, valid3.astype(float), bounds_error=False, fill_value=0,
        method="nearest")

    # 3D density (for grad ln rho alignment test)
    mn = [e[0] for e in edges]
    mx = [e[-1] for e in edges]
    H3, _ = np.histogramdd(Z, bins=args.bins,
                           range=[[mn[j], mx[j]] for j in range(3)])
    D3 = gaussian_filter(H3, sigma=1.5)
    D3 /= max(D3.max(), 1e-12)
    logD3 = np.log(D3 + 1e-9)
    g3 = np.gradient(logD3)          # list of 3 arrays, axis order (0,1,2)
    grads = np.stack([g3[0], g3[1], g3[2]], axis=-1)   # (B,B,B,3)
    grad3_interp = RegularGridInterpolator(
        centers, grads, bounds_error=False, fill_value=np.nan)

    # ---- 3D diagnostics over the whole volume (valid bins) ----
    with np.errstate(invalid="ignore", divide="ignore"):
        div3 = (np.gradient(b3_f[..., 0], centers[0], axis=0)
                + np.gradient(b3_f[..., 1], centers[1], axis=1)
                + np.gradient(b3_f[..., 2], centers[2], axis=2))
        curl3 = np.stack([
            np.gradient(b3_f[..., 2], centers[1], axis=1)
            - np.gradient(b3_f[..., 1], centers[2], axis=2),
            np.gradient(b3_f[..., 0], centers[2], axis=2)
            - np.gradient(b3_f[..., 2], centers[0], axis=0),
            np.gradient(b3_f[..., 1], centers[0], axis=0)
            - np.gradient(b3_f[..., 0], centers[1], axis=1)], axis=-1)
    curl3_n = np.linalg.norm(curl3, axis=-1)
    b3_n = np.linalg.norm(b3_f, axis=-1)
    gn3_n = np.linalg.norm(grads, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos3 = (b3_f * grads).sum(-1) / (b3_n * gn3_n)

    cntv = cnt0[valid3]
    med = np.median(cntv)
    dense = valid3 & (cnt0 >= med)
    sparse = valid3 & (cnt0 < med)
    print("\n=== 3D PCA-space verdicts (whole volume) ===")
    print(f"mean div, dense bins  = {np.nanmean(div3[dense]):.3f} /s")
    print(f"mean div, sparse bins = {np.nanmean(div3[sparse]):.3f} /s")
    good = valid3 & (b3_n > np.nanpercentile(b3_n[valid3], 50)) & (gn3_n > 1e-3)
    print(f"mean cos(b, grad ln rho), strong-flow bins = "
          f"{np.nanmean(np.where(good, cos3, np.nan)):.3f}")
    print(f"mean |curl| dense = {np.nanmean(curl3_n[dense]):.3f} /s, "
          f"sparse = {np.nanmean(curl3_n[sparse]):.3f} /s")

    # ---- 3D diagnostics over the zoom volume (points in UMAP window) ----
    in_win = (emb[:, 0] >= win[0]) & (emb[:, 0] <= win[1]) & \
             (emb[:, 1] >= win[2]) & (emb[:, 1] <= win[3])
    Zw = Z[in_win]
    vol_lo = Zw.min(axis=0) - 0.5 * np.ptp(Zw, axis=0) * 0.1
    vol_hi = Zw.max(axis=0) + 0.5 * np.ptp(Zw, axis=0) * 0.1
    mgrid = np.meshgrid(*centers, indexing="ij")
    in_vol = ((mgrid[0] >= vol_lo[0]) & (mgrid[0] <= vol_hi[0]) &
              (mgrid[1] >= vol_lo[1]) & (mgrid[1] <= vol_hi[1]) &
              (mgrid[2] >= vol_lo[2]) & (mgrid[2] <= vol_hi[2]))
    sel = in_vol & valid3
    if sel.any():
        cntv2 = cnt0[sel]
        med2 = np.median(cntv2)
        d2, s2 = sel & (cnt0 >= med2), sel & (cnt0 < med2)
        print(f"\n=== zoom volume ({sel.sum()} valid bins) ===")
        print(f"mean div, dense  = {np.nanmean(div3[d2]):.3f} /s")
        print(f"mean div, sparse = {np.nanmean(div3[s2]):.3f} /s")

    # ================= 2D reference field =================
    from plot_vortex import (drift_ladder, extrapolate_drift, gap_fill,
                             _bin_means)  # local reuse
    xc, yc, fields2, cnt02 = drift_ladder(emb, dt, ladder)
    b2 = gap_fill(extrapolate_drift(fields2, ladder))
    valid2 = cnt02 >= 10
    X2, Y2 = np.meshgrid(xc, yc)
    U2 = np.ma.masked_where(~valid2, b2[..., 0])
    V2 = np.ma.masked_where(~valid2, b2[..., 1])
    div2 = (np.gradient(b2[..., 0], xc, axis=1)
            + np.gradient(b2[..., 1], yc, axis=0))

    # ---- zoom slices for display ----
    def sl2(arr1d, lo, hi):
        i0 = int(np.clip(np.searchsorted(arr1d, lo) - 1, 0, len(arr1d) - 1))
        i1 = int(np.clip(np.searchsorted(arr1d, hi) + 1, 1, len(arr1d)))
        return i0, i1

    ix0, ix1 = sl2(xc, win[0], win[1])
    iy0, iy1 = sl2(yc, win[2], win[3])
    id0, id1 = sl2(xs_d, win[0], win[1])
    jd0, jd1 = sl2(ys_d, win[2], win[3])
    Dz = D2d[jd0:jd1, id0:id1]
    ext_z = [xs_d[id0], xs_d[id1 - 1], ys_d[jd0], ys_d[jd1 - 1]]
    X2z = X2[iy0:iy1, ix0:ix1]
    Y2z = Y2[iy0:iy1, ix0:ix1]
    U2z = U2[iy0:iy1, ix0:ix1]
    V2z = V2[iy0:iy1, ix0:ix1]
    div2z = div2[iy0:iy1, ix0:ix1]

    # ================= integrate b3 in 3D, project to UMAP =================
    n_seed = 13
    sg = np.linspace(win[0] + 0.15 * w, win[1] - 0.15 * w, n_seed)
    tsg = np.linspace(win[2] + 0.15 * w, win[3] - 0.15 * w, n_seed)
    SX, SY = np.meshgrid(sg, tsg)
    seeds_emb = np.stack([SX.ravel(), SY.ravel()], axis=1)
    # drop seeds far from any data
    dd, _ = lmap.tree_emb.query(seeds_emb, k=1)
    keep = dd.ravel() < 3 * lmap.h_emb
    seeds_emb = seeds_emb[keep]
    seeds_Z = lmap.emb_to_pca(seeds_emb)

    vclip = np.nanpercentile(b3_n[valid3], 95)
    print(f"\nintegrating {len(seeds_emb)} seeds in PCA space "
          f"(clip |b|={vclip:.2f}/s) ...")
    h_step, n_step = 0.05, 260        # 13 s of flow
    paths_emb = []
    for z0 in seeds_Z:
        z = z0.copy()
        pts = []
        for _ in range(n_step):
            zz = z[None, :]
            if not valid3_interp(zz)[0] > 0.5:
                break
            bz = b3_interp(zz)[0]
            if np.isnan(bz).any():
                break
            nrm = np.linalg.norm(bz)
            if nrm < 1e-9:
                break
            if nrm > vclip:
                bz = bz / nrm * vclip
            z = z + bz * h_step
            pts.append(z.copy())
        if len(pts) >= 4:
            P = np.array(pts)
            paths_emb.append(lmap.pca_to_emb(P))
    print(f"{len(paths_emb)} integrated paths survived validity mask")

    # divergence of b3 sampled over the 2D zoom grid -> projected display
    grid_emb = np.stack([X2z.ravel(), Y2z.ravel()], axis=1)
    grid_Z = lmap.emb_to_pca(grid_emb)
    div_proj = RegularGridInterpolator(
        centers, div3, bounds_error=False, fill_value=np.nan
    )(grid_Z).reshape(X2z.shape)
    print(f"projected 3D divergence: "
          f"{np.isfinite(div_proj).sum()} valid cells of {div_proj.size}")

    # ================= panel B: real dwell runs =================
    runs = window_runs(emb, win)
    print(f"{len(runs)} dwell runs (>=1s) through window")
    t0 = time_ds[0]
    tspan = max(time_ds[-1] - time_ds[0], 1e-9)

    # ================= figure =================
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # A: 2D reference
    ax = axes[0]
    ax.contourf(Dz, extent=ext_z, levels=12, cmap="Greys", alpha=0.55)
    ax.contour(Dz, extent=ext_z, levels=6, colors="k", linewidths=0.4,
               alpha=0.5)
    ax.streamplot(X2z, Y2z, U2z, V2z, color="tab:red", density=2.2,
                  linewidth=1.0, arrowsize=1.2)
    ax.plot(px, py, marker="*", ms=18, mfc="gold", mec="k")
    ax.set_title("A. reference: b(x) estimated in 2D UMAP space\n"
                 "(distorted by projection folding)")

    # B: 3D-integrated paths projected back
    ax = axes[1]
    ax.contour(Dz, extent=ext_z, levels=8, colors="k", linewidths=0.4,
               alpha=0.45)
    for P in paths_emb:
        segs = np.stack([P[:-1], P[1:]], axis=1)
        lc = LineCollection(segs, colors=["tab:blue"], linewidths=1.1,
                            alpha=0.8)
        ax.add_collection(lc)
        for q in (len(P) // 4, len(P) // 2, 3 * len(P) // 4):
            if q >= 1:
                ax.annotate("", xy=P[q], xytext=P[q - 1],
                            arrowprops=dict(arrowstyle="-|>",
                                            color="tab:blue",
                                            lw=0.8, mutation_scale=8))
    # real dwell runs, faint, for context
    for (s, e) in runs[:args.max_runs]:
        pts = emb[s:e]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        tc = (time_ds[s:e - 1] - t0) / tspan
        ax.add_collection(LineCollection(segs, array=tc, cmap="viridis",
                                         linewidths=0.6, alpha=0.35))
    ax.plot(px, py, marker="*", ms=18, mfc="gold", mec="k")
    ax.set_title("B. b(x) estimated in 3D PCA, integrated there,\n"
                 "projected to UMAP (blue = model flow; faint = real runs)")

    # C: projected 3D divergence
    ax = axes[2]
    lim = np.nanpercentile(np.abs(div_proj[np.isfinite(div_proj)]), 95)
    lim = max(lim, 1e-6)
    cf = ax.contourf(X2z, Y2z, np.clip(div_proj, -lim, lim), levels=20,
                     cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(cf, ax=ax, label="div b  (1/s)")
    ax.contour(Dz, extent=ext_z, levels=8, colors="k", linewidths=0.4,
               alpha=0.4)
    ax.plot(px, py, marker="*", ms=18, mfc="gold", mec="k")
    ax.set_title("C. 3D divergence of b, projected\n(red = outflow, blue = inflow)")

    for ax in axes:
        ax.set_xlim(win[0], win[1])
        ax.set_ylim(win[2], win[3])
        ax.set_aspect("equal")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    fig.suptitle(f"{recording} | 3D-PCA drift, peak ({px:.2f}, {py:.2f}) "
                 f"window +/-{w}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, str(latent_dir / f"{recording}_vortex_zoom_pca3d.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
