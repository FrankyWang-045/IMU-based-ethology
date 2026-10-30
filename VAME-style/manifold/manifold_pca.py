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

from utils import load_config, output_dir


def plot_manifold_umap(
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
):
    """
    对单个 recording 的 6 维 merged 数据做时段截取 + UMAP 降维 + 时序散点可视化。
    默认输出 1600×900 像素，适配 1920×1080 屏幕。
    """
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

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    coords = reducer.fit_transform(segment)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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
    ax.set_title(f"{recording_name} | t={t_start:.1f}s-{t_end:.1f}s")
    ax.legend(loc="best")
    plt.colorbar(scatter, ax=ax, label="time (s)")

    save_name = f"{recording_name}_{t_start:.1f}_{t_end:.1f}.png"
    save_path = in_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")
    print(f"  样本数: {len(segment)}, 图片尺寸: {figsize[0]*dpi:.0f}×{figsize[1]*dpi:.0f}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="6D -> 2D UMAP 流形时序可视化")
    parser.add_argument("--recording", required=True, help="recording 名（不含 .npz）")
    parser.add_argument("--start", type=float, required=True, help="起始时间戳（秒）")
    parser.add_argument("--end", type=float, required=True, help="结束时间戳（秒）")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    cfg = load_config()
    plot_manifold_umap(
        cfg,
        args.recording,
        args.start,
        args.end,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )


if __name__ == "__main__":
    main()