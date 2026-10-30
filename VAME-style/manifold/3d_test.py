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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from utils import load_config, output_dir


def add_time_feature(features, time, time_weight=1.0):
    if time_weight <= 0:
        return features
    t_min, t_max = time.min(), time.max()
    t_norm = (time - t_min) / (t_max - t_min + 1e-8)
    t_norm = t_norm[:, None] * time_weight
    return np.hstack([features, t_norm])


def plot_manifold_umap_3d(
    cfg,
    recording_name,
    t_start,
    t_end,
    input_subdir="direction_gravity",
    cmap="viridis",
    figsize=(16, 9),
    dpi=100,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    time_weight=0.0,
    max_points=20000,
):
    out = output_dir(cfg)
    in_dir = out / input_subdir
    npz_path = in_dir / f"{recording_name}.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到文件: {npz_path}")

    data = np.load(npz_path)
    features = data["features"]  # [T, 6]
    time = data["time"]

    mask = (time >= t_start) & (time <= t_end)
    if not mask.any():
        raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无数据")

    segment = features[mask]
    segment_time = time[mask]

    # 限制最大点数
    if len(segment) > max_points:
        idx = np.linspace(0, len(segment) - 1, max_points, dtype=int)
        segment = segment[idx]
        segment_time = segment_time[idx]
        print(f"  点数超过 {max_points}，已均匀降采样至 {max_points}")

    segment_for_umap = add_time_feature(segment, segment_time, time_weight)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    coords = reducer.fit_transform(segment_for_umap)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(segment_time.min(), segment_time.max())

    # ---------- 3D 时序连线 ----------
    segments = np.stack([coords[:-1], coords[1:]], axis=1)  # [N-1, 2, 3]
    lc = Line3DCollection(
        segments,
        cmap=cmap,
        norm=norm,
        alpha=0.4,
        linewidths=0.6,
    )
    lc.set_array(segment_time[:-1])
    ax.add_collection3d(lc)

    # ---------- 3D 散点 ----------
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=segment_time,
        cmap=cmap,
        s=15,
        alpha=0.8,
        edgecolors="none",
        zorder=3,
    )

    # 起点 / 终点
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        coords[0, 2],
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
        coords[-1, 2],
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
    ax.set_zlabel("UMAP 3")
    ax.set_title(
        f"{recording_name} | t={t_start:.1f}s-{t_end:.1f}s | "
        f"time_weight={time_weight} | n={len(segment)}"
    )

    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="time (s)")

    save_name = f"{recording_name}_{t_start:.1f}_{t_end:.1f}_3d_tw{time_weight}.png"
    save_path = in_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")
    print(f"  样本数: {len(segment)}, UMAP 输入维度: {segment_for_umap.shape[1]}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="6D -> 3D UMAP 流形时序可视化")
    parser.add_argument("--recording", required=True, help="recording 名（不含 .npz）")
    parser.add_argument("--start", type=float, required=True, help="起始时间戳（秒）")
    parser.add_argument("--end", type=float, required=True, help="结束时间戳（秒）")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--time_weight", type=float, default=0.0, help="时间加权系数")
    parser.add_argument("--max_points", type=int, default=20000, help="最大点数限制")
    args = parser.parse_args()

    cfg = load_config()
    plot_manifold_umap_3d(
        cfg,
        args.recording,
        args.start,
        args.end,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        time_weight=args.time_weight,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()