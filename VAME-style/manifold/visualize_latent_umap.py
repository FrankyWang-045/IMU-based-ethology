from pathlib import Path
import argparse

# ---------- 修复 umap-learn 与新版 scikit-learn 兼容性 ----------
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
# --------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.collections import LineCollection

from utils import load_config, output_dir


def plot_latent_umap(
    cfg,
    recording_name,
    t_start,
    t_end,
    latent_subdir="latent",
    cmap="viridis",
    figsize=(16, 9),
    dpi=100,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    max_points=20000,
):
    out = output_dir(cfg)
    latent_dir = out / latent_subdir
    npz_path = latent_dir / f"{recording_name}_latent.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到文件: {npz_path}")

    data = np.load(npz_path)
    latent = data["latent"]
    time = data["time"]

    mask = (time >= t_start) & (time <= t_end)
    if not mask.any():
        raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无数据")

    segment = latent[mask]
    segment_time = time[mask]

    if len(segment) > max_points:
        idx = np.linspace(0, len(segment) - 1, max_points, dtype=int)
        segment = segment[idx]
        segment_time = segment_time[idx]
        print(f"  点数超过 {max_points}，已均匀降采样至 {max_points}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    coords = reducer.fit_transform(segment)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    norm = plt.Normalize(segment_time.min(), segment_time.max())

    # ---------- 时序连线 ----------
    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        alpha=0.5,
        linewidths=0.8,
    )
    lc.set_array(segment_time[:-1])
    ax.add_collection(lc)

    # ---------- 散点 ----------
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=segment_time,
        cmap=cmap,
        s=15,
        alpha=0.8,
        edgecolors="none",
        zorder=3,
    )

    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        c="lime",
        s=150,
        marker="o",
        edgecolors="black",
        linewidths=1.5,
        label="start",
        zorder=5,
    )
    ax.scatter(
        coords[-1, 0],
        coords[-1, 1],
        c="red",
        s=150,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="end",
        zorder=5,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        f"{recording_name} latent | t={t_start:.1f}s-{t_end:.1f}s | n={len(segment)}"
    )
    ax.legend(loc="best")
    plt.colorbar(scatter, ax=ax, label="time (s)")

    save_name = f"{recording_name}_latent_umap_{t_start:.1f}_{t_end:.1f}.png"
    save_path = latent_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")
    print(f"  样本数: {len(segment)}, latent dim: {latent.shape[1]}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="latent UMAP 2D 时序可视化")
    parser.add_argument("--recording", required=True, help="recording 名（不含 _latent.npz）")
    parser.add_argument("--start", type=float, required=True, help="起始时间戳（秒）")
    parser.add_argument("--end", type=float, required=True, help="结束时间戳（秒）")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--max_points", type=int, default=20000, help="最大点数限制")
    args = parser.parse_args()

    cfg = load_config()
    plot_latent_umap(
        cfg,
        args.recording,
        args.start,
        args.end,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()