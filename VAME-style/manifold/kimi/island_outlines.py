"""
Island outline extraction in UMAP display space.

For each density peak that survives a persistence threshold, extract the
level set  rho = max(frac * peak_height, h_abs)  restricted to the peak's
catchment basin (from watershed.py flood_persistence), then simplify
(Ramer-Douglas-Peucker) and smooth (circular moving average) the contour
into a hand-drawn-style closed outline.

Outputs (next to the latent npz):
  <rec>_islands.png     terrain + faint scatter + colored outlines
  <rec>_islands.svg     vector version (contours + axes only)
  <rec>_islands.json    readable contour coordinates + per-island metadata
  <rec>_islands.npz     archive: label grid, density grid, raw contours

Usage:
  python island_outlines.py --latent_path <...>_latent.npz \
      [--h 0.005] [--frac 0.3] [--h_abs 0.005] [--downsample 4]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from watershed import flood_persistence  # noqa: E402
from plot_vortex import load_or_make_embedding, density_grid, save_fig  # noqa: E402


# ------------------------------------------------- contour post-processing
def rdp(points, eps):
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    nrm = np.hypot(*ab)
    if nrm < 1e-12:
        d = np.sqrt(((points - a) ** 2).sum(axis=1))
    else:
        d = np.abs(np.cross(ab, points - a)) / nrm
    i = int(np.argmax(d))
    if d[i] <= eps:
        return np.vstack([a, b])
    left = rdp(points[:i + 1], eps)
    right = rdp(points[i:], eps)
    return np.vstack([left[:-1], right])


def smooth_closed(points, win=7, passes=3):
    """Circular moving-average smoothing for a closed outline."""
    k = np.ones(win) / win
    out = points.copy()
    n = len(out)
    pad = win // 2
    for _ in range(passes):
        xs = np.r_[out[-pad:, 0], out[:, 0], out[:pad, 0]]
        ys = np.r_[out[-pad:, 1], out[:, 1], out[:pad, 1]]
        out = np.stack([np.convolve(xs, k, "valid"),
                        np.convolve(ys, k, "valid")], axis=1)
        assert len(out) == n
    return out


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def extract_contour(D, xs, ys, basin_mask, level, peak_rc):
    """Closed outline of the island: the connected component of
    {D >= level} inside the basin that contains the peak cell.
    Returns (N,2) array in data coords."""
    from scipy import ndimage
    M = (D >= level) & basin_mask
    cc, _ = ndimage.label(M)
    pl = cc[peak_rc]
    if pl == 0:
        return None
    C = (cc == pl).astype(float)
    # pad with a zero ring so the 0.5-level loop is closed and inside grid
    Cp = np.pad(C, 1)
    dxs = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dys = (ys[1] - ys[0]) if len(ys) > 1 else 1.0
    xsp = np.concatenate([[xs[0] - dxs], xs, [xs[-1] + dxs]])
    ysp = np.concatenate([[ys[0] - dys], ys, [ys[-1] + dys]])
    X, Y = np.meshgrid(xsp, ysp)
    fig = plt.figure()
    cs = plt.contour(X, Y, Cp, levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) >= 8]
    plt.close(fig)
    if not segs:
        return None
    best = max(segs, key=len)
    if np.hypot(*(best[0] - best[-1])) < 1e-6:
        best = best[:-1]
    return best


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Island outlines in UMAP space")
    ap.add_argument("--latent_path", required=True)
    ap.add_argument("--h", type=float, default=0.005,
                    help="persistence threshold for keeping peaks")
    ap.add_argument("--min_peak", type=float, default=0.1,
                    help="absolute peak-height threshold (density is 0..1)")
    ap.add_argument("--level", type=float, default=0.1,
                    help="absolute density level for island outlines "
                         "(density is normalized 0..1); basin-restricted")
    ap.add_argument("--eps_cells", type=float, default=1.5,
                    help="RDP simplification tolerance in grid cells")
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="density smoothing, in grid cells (fatter mountains)")
    ap.add_argument("--smooth_win", type=int, default=7)
    ap.add_argument("--downsample", type=int, default=4)
    args = ap.parse_args()

    latent_path = Path(args.latent_path)
    latent_dir = latent_path.parent
    recording = latent_path.name.replace("_latent.npz", "")

    d = np.load(latent_path)
    latent = d["latent"].astype(np.float64)
    latent_ds = np.ascontiguousarray(latent[::args.downsample])

    emb = load_or_make_embedding(latent_dir, recording, latent_ds)
    D, extent = density_grid(emb, res=200, sigma=args.sigma)
    xs = np.linspace(extent[0], extent[1], D.shape[1])
    ys = np.linspace(extent[2], extent[3], D.shape[0])
    cell = (extent[1] - extent[0]) / D.shape[1]
    eps = args.eps_cells * cell
    print(f"density grid {D.shape}, cell size {cell:.4f}, eps={eps:.4f}")

    # ---------------- peaks + catchment basins ----------------
    # flood_persistence 的 cell_peak 只记录"零水位连通域"的最终幸存峰，
    # 不是按峰分割的流域；流域用 marker-based watershed(-D) 重新划分。
    t0 = time.time()
    peaks_flat, peak_h, persist, killer, _ = flood_persistence(D)
    print(f"flood: {len(peak_h)} candidate peaks in {time.time()-t0:.1f}s")

    print(f"persistence h={args.h}: "
          f"{int((np.isfinite(persist) & (persist >= args.h)).sum()) + 1} "
          f"peaks survive")
    # h-scan table to help tuning
    fin = persist[np.isfinite(persist)]
    row = "  h-scan: " + "  ".join(
        f"h={h:g}:{int((fin >= h).sum()) + 1}"
        for h in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1))
    print(row)

    # "sufficiently high peaks": persistence AND absolute height floors
    keep_peaks = [p for p in range(len(persist))
                  if persist[p] >= args.h and peak_h[p] >= args.min_peak]
    keep_peaks.sort(key=lambda p: -persist[p])
    n_keep = len(keep_peaks)
    print(f"after min_peak={args.min_peak}: {n_keep} peaks remain")
    if n_keep == 0:
        raise SystemExit("no peaks survive; lower --h or --min_peak")

    # catchment basins: markers at kept-peak cells, flood on -D
    from skimage.segmentation import watershed
    markers = np.zeros(D.shape, dtype=int)
    for k, p in enumerate(keep_peaks, start=1):
        iy, ix = np.unravel_index(peaks_flat[p], D.shape)
        markers[iy, ix] = k
    labels2d = watershed(-D, markers=markers, mask=D > 0)

    # ---------------- per-island contour ----------------
    islands = []
    raw_contours = {}
    cmap = plt.get_cmap("tab20")
    for k, p in enumerate(keep_peaks, start=1):
        iid = k - 1
        iy, ix = np.unravel_index(peaks_flat[p], D.shape)
        px, py = float(xs[ix]), float(ys[iy])
        height = float(peak_h[p])
        per = float(persist[p]) if np.isfinite(persist[p]) else None
        # one absolute water level for all islands; the watershed basin
        # keeps neighboring islands separate at low levels
        level = args.level
        basin = labels2d == iid + 1   # watershed markers were 1-based
        if basin.sum() < 20:
            print(f"  island {iid}: basin too small ({basin.sum()} cells), skip")
            continue
        cont = None
        lv = level
        for _attempt in range(4):
            cont = extract_contour(D, xs, ys, basin, lv, (iy, ix))
            if cont is not None:
                break
            lv *= 0.85          # retry slightly lower
        if cont is None:
            print(f"  island {iid}: no closed contour near level {level:.4f}")
            continue
        if lv != level:
            print(f"  island {iid}: level relaxed {level:.4f} -> {lv:.4f}")
            level = lv
        raw_contours[str(iid)] = cont
        # smooth FIRST on the dense raw contour (moving-average on a dense
        # closed ring barely shrinks it), THEN RDP-simplify; reversing the
        # order collapses small polygons toward their centroid
        sm = smooth_closed(cont, win=args.smooth_win, passes=2)
        simp = rdp(np.vstack([sm, sm[0]]), eps)
        if len(simp) > 8:
            simp = simp[:-1]
        area = polygon_area(simp)
        islands.append({
            "id": iid,
            "peak": [round(px, 4), round(py, 4)],
            "height": round(height, 5),
            "persistence": (round(per, 5) if per is not None else "inf"),
            "level": round(level, 5),
            "area": round(area, 4),
            "n_vertices": int(len(simp)),
            "contour": [[round(float(x), 4), round(float(y), 4)]
                        for x, y in simp],
        })
        print(f"  island {iid}: peak=({px:.2f},{py:.2f}) h={height:.4f} "
              f"persist={per if per is None else round(per,4)} "
              f"level={level:.4f} area={area:.2f} verts={len(simp)}")

    # ---------------- figure ----------------
    sel = np.arange(len(emb))
    if len(sel) > 40000:
        sel = np.random.default_rng(0).choice(len(emb), 40000, replace=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(D, extent=extent, levels=15, cmap="terrain", alpha=0.35)
    ax.scatter(emb[sel, 0], emb[sel, 1], s=1, c="0.7", alpha=0.4,
               linewidths=0)
    for isl in islands:
        c = np.array(isl["contour"])
        col = cmap(isl["id"] % 20)
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]],
                color=col, lw=2.2)
        ax.plot(*isl["peak"], marker="*", ms=16, mfc=col, mec="k")
        ax.annotate(str(isl["id"]), isl["peak"], textcoords="offset points",
                    xytext=(8, 8), fontsize=11, weight="bold", color=col)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{recording} | island outlines "
                 f"(persistence >= {args.h}, min_peak >= {args.min_peak}, "
                 f"level = {args.level})")
    save_fig(fig, str(latent_dir / f"{recording}_islands.png"))
    plt.close(fig)

    # ---------------- SVG (contours + axes only) ----------------
    svg_path = os.path.normpath(str(latent_dir / f"{recording}_islands.svg"))
    W, H = 800, 640
    sx = (W - 80) / (extent[1] - extent[0])
    sy = (H - 80) / (extent[3] - extent[2])
    sc = min(sx, sy)
    def to_px(x, y):
        return (40 + (x - extent[0]) * sc,
                H - 40 - (y - extent[2]) * sc)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
             f'height="{H}" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<line x1="40" y1="{H-40}" x2="{W-20}" y2="{H-40}" stroke="black"/>',
             f'<line x1="40" y1="{H-40}" x2="40" y2="20" stroke="black"/>']
    arrow = '<defs><marker id="ar" markerWidth="10" markerHeight="10" ' \
            'refX="8" refY="3" orient="auto">' \
            '<path d="M0,0 L8,3 L0,6 z" fill="black"/></marker></defs>'
    parts.insert(1, arrow)
    parts[2] = parts[2][:-1] + ' marker-end="url(#ar)"/>'
    parts[3] = parts[3][:-1] + ' marker-end="url(#ar)"/>'
    for isl in islands:
        col = matplotlib.colors.to_hex(cmap(isl["id"] % 20))
        pts = " ".join(f"{to_px(x, y)[0]:.1f},{to_px(x, y)[1]:.1f}"
                       for x, y in isl["contour"])
        parts.append(f'<polygon points="{pts}" fill="none" stroke="{col}" '
                     f'stroke-width="2.5" stroke-linejoin="round"/>')
        px_, py_ = to_px(*isl["peak"])
        parts.append(f'<text x="{px_+8:.1f}" y="{py_-8:.1f}" fill="{col}" '
                     f'font-size="16" font-weight="bold">{isl["id"]}</text>')
    parts.append("</svg>")
    for attempt in range(3):
        try:
            with open(svg_path, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
            print(f"Saved: {svg_path}")
            break
        except OSError as e:
            if attempt == 2:
                raise
            print(f"svg save failed ({e}), retry {attempt + 1}/3 ...")
            time.sleep(1)

    # ---------------- JSON + npz ----------------
    payload = {
        "recording": recording,
        "space": "umap_display",
        "params": {"persistence_h": args.h, "min_peak": args.min_peak,
                   "level": args.level, "rdp_eps_cells": args.eps_cells,
                   "density_sigma": args.sigma,
                   "smooth_win": args.smooth_win,
                   "density_grid_res": 200},
        "extent": [round(v, 4) for v in extent],
        "n_islands": len(islands),
        "islands": islands,
    }
    json_path = os.path.normpath(str(latent_dir / f"{recording}_islands.json"))
    for attempt in range(3):
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            print(f"Saved: {json_path}")
            break
        except OSError as e:
            if attempt == 2:
                raise
            print(f"json save failed ({e}), retry {attempt + 1}/3 ...")
            time.sleep(1)

    np.savez_compressed(
        latent_dir / f"{recording}_islands.npz",
        labels=labels2d, D=D,
        raw_contours=np.array([raw_contours[str(i["id"])] for i in islands],
                              dtype=object),
        island_ids=np.array([i["id"] for i in islands]),
        extent=np.array(extent))
    print(f"Saved: {latent_dir / (recording + '_islands.npz')}")


if __name__ == "__main__":
    main()
