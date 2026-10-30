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
from scipy.ndimage import gaussian_filter

from utils import load_config, output_dir


def load_or_make_embedding(latent_dir, recording, latent):
    """
    统一坐标系嵌入（与 plot_ws_hierarchy.py 共用）：
    UMAP(n_neighbors=15, min_dist=0.1, random_state=42)，
    输入为与 islands npz 对齐的 [::4] 分析帧。
    首次运行计算并保存 <recording>_embedding.npy，之后所有脚本复用，
    保证各图坐标系完全一致。
    """
    p = Path(latent_dir) / f"{recording}_embedding.npy"
    if p.exists():
        print(f"使用共享嵌入: {p}")
        return np.load(p)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(latent)
    np.save(p, emb)
    return emb


def plot_umap_terrain(
    cfg,
    recording_name,
    t_start=None,
    t_end=None,
    latent_subdir="latent",
    cmap="terrain",
    figsize=(14, 10),
    dpi=150,
    bins=100,
    sigma=2.0,
    n_neighbors=15,
    min_dist=0.1,
    max_points=50000,
    elev=30,
    azim=45,
):
    """
    对 latent 做 UMAP 降维，计算 2D 密度，绘制三维地形图。
    密度越高，地形高度越高，颜色按高度映射。
    """
    out = output_dir(cfg)
    latent_dir = out / latent_subdir
    npz_path = latent_dir / f"{recording_name}_latent.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到文件: {npz_path}")

    data = np.load(npz_path)
    latent = data["latent"]
    time = data["time"]

    # 统一坐标系：嵌入定义在 [::4] 分析帧集上（与 islands npz 的
    # states / time / frame_density 逐帧对齐），两个脚本共用同一文件。
    # 注意：n_neighbors / min_dist / max_points 参数现由共享嵌入文件固定，
    # 传入参数仅保留以兼容旧命令行。
    latent_ds = latent[::4]
    time_ds = time[::4]
    coords = load_or_make_embedding(latent_dir, recording_name, latent_ds)

    # 时间截取（可选）
    if t_start is not None and t_end is not None:
        mask = (time_ds >= t_start) & (time_ds <= t_end)
        if not mask.any():
            raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无数据")
        segment = coords[mask]
        title_suffix = f" | t={t_start:.1f}s-{t_end:.1f}s"
    else:
        segment = coords
        title_suffix = ""

    print(f"统一嵌入坐标: {segment.shape[0]} 点 (UMAP 15/0.1/rs42)")

    # 2D 密度直方图 + 高斯平滑（对当前段/全量计算，与嵌入同一坐标系）
    H, xedges, yedges = np.histogram2d(
        segment[:, 0], segment[:, 1], bins=bins
    )
    H_smooth = gaussian_filter(H, sigma=sigma)

    # 构建网格
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    Z = H_smooth.T

    # 绘制 3D 地形
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("density")
    ax.set_title(f"{recording_name} | UMAP density terrain{title_suffix}")

    # 视角
    ax.view_init(elev=elev, azim=azim)

    # 颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("density")

    # 保存
    if t_start is not None and t_end is not None:
        save_name = f"{recording_name}_umap_terrain_{t_start:.1f}_{t_end:.1f}.png"
    else:
        save_name = f"{recording_name}_umap_terrain.png"
    save_path = latent_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="UMAP 三维地形密度图")
    parser.add_argument("--recording", required=True, help="recording 名")
    parser.add_argument("--start", type=float, default=None, help="起始时间戳（秒，可选）")
    parser.add_argument("--end", type=float, default=None, help="结束时间戳（秒，可选）")
    parser.add_argument("--bins", type=int, default=100, help="密度网格数")
    parser.add_argument("--sigma", type=float, default=2.0, help="高斯平滑 sigma")
    parser.add_argument("--cmap", type=str, default="terrain", help="高度 colormap")
    parser.add_argument("--elev", type=float, default=30, help="俯仰角")
    parser.add_argument("--azim", type=float, default=45, help="方位角")
    parser.add_argument("--max_points", type=int, default=50000, help="最大点数限制")
    args = parser.parse_args()

    cfg = load_config()
    plot_umap_terrain(
        cfg,
        args.recording,
        t_start=args.start,
        t_end=args.end,
        bins=args.bins,
        sigma=args.sigma,
        cmap=args.cmap,
        elev=args.elev,
        azim=args.azim,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()