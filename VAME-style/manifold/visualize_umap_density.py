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
from scipy.stats import gaussian_kde

from utils import load_config, output_dir


def plot_umap_density(
    cfg,
    recording_name,
    t_start=None,
    t_end=None,
    latent_subdir="latent",
    cmap="hot",
    figsize=(16, 9),
    dpi=100,
    n_neighbors=15,
    min_dist=0.1,
    grid_size=200,
    levels=20,
    max_points=50000,
    overlay_scatter=True,
):
    """
    对 latent 做 UMAP 降维后，绘制 2D 密度分布图。
    密度越高的区域颜色越深。
    """
    out = output_dir(cfg)
    latent_dir = out / latent_subdir
    npz_path = latent_dir / f"{recording_name}_latent.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到文件: {npz_path}")

    data = np.load(npz_path)
    latent = data["latent"]
    time = data["time"]

    # 时间截取（可选）
    if t_start is not None and t_end is not None:
        mask = (time >= t_start) & (time <= t_end)
        if not mask.any():
            raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无数据")
        segment = latent[mask]
        title_suffix = f" | t={t_start:.1f}s-{t_end:.1f}s"
    else:
        segment = latent
        title_suffix = ""

    # 限制最大点数
    if len(segment) > max_points:
        idx = np.linspace(0, len(segment) - 1, max_points, dtype=int)
        segment = segment[idx]
        print(f"  点数超过 {max_points}，已均匀降采样至 {max_points}")

    print(f"UMAP 降维: {segment.shape[0]} 样本 x {segment.shape[1]} 维")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(segment)

    # 计算 2D KDE
    print("计算 2D 密度估计...")
    kde = gaussian_kde(coords.T)

    # 构建网格
    margin = 0.05
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = (x_max - x_min) * margin + 1e-6
    y_pad = (y_max - y_min) * margin + 1e-6

    xx, yy = np.mgrid[
        x_min - x_pad : x_max + x_pad : complex(grid_size),
        y_min - y_pad : y_max + y_pad : complex(grid_size),
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)

    # 绘图
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 填充等高线/密度热力图
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap=cmap)

    # 可选：叠加原始散点
    if overlay_scatter:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="white",
            s=2,
            alpha=0.2,
            edgecolors="none",
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{recording_name} | UMAP density{title_suffix}")

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("density")

    # 保存
    if t_start is not None and t_end is not None:
        save_name = f"{recording_name}_umap_density_{t_start:.1f}_{t_end:.1f}.png"
    else:
        save_name = f"{recording_name}_umap_density.png"
    save_path = latent_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="UMAP 空间密度可视化")
    parser.add_argument("--recording", required=True, help="recording 名")
    parser.add_argument("--start", type=float, default=None, help="起始时间戳（秒，可选）")
    parser.add_argument("--end", type=float, default=None, help="结束时间戳（秒，可选）")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--cmap", type=str, default="hot", help="colormap，例如 hot / magma / viridis")
    parser.add_argument("--levels", type=int, default=20, help="密度等高线层数")
    parser.add_argument("--max_points", type=int, default=50000, help="最大点数限制")
    args = parser.parse_args()

    cfg = load_config()
    plot_umap_density(
        cfg,
        args.recording,
        t_start=args.start,
        t_end=args.end,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        cmap=args.cmap,
        levels=args.levels,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()