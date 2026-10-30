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

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from utils import load_config, output_dir


def render_frame(coords, current_idx, window_size, figsize, dpi):
    """绘制单帧：滑动窗口 + 尾部淡出。"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    start_idx = max(0, current_idx - window_size)
    visible = coords[start_idx:current_idx]

    if len(visible) > 1:
        rel = np.linspace(0, 1, len(visible))

        segments = np.stack([visible[:-1], visible[1:]], axis=1)
        alphas = rel[1:] ** 2
        colors = np.zeros((len(segments), 4))
        colors[:, 0] = 0.27  # R steelblue
        colors[:, 1] = 0.51  # G
        colors[:, 2] = 0.71  # B
        colors[:, 3] = 0.1 + 0.6 * alphas

        lc = LineCollection(
            segments,
            colors=colors,
            linewidths=1.5,
        )
        ax.add_collection(lc)

        point_alphas = 0.15 + 0.75 * (rel ** 2)
        point_sizes = 10 + 30 * rel
        for i in range(len(visible)):
            ax.scatter(
                visible[i, 0],
                visible[i, 1],
                c="steelblue",
                s=point_sizes[i],
                alpha=point_alphas[i],
                edgecolors="none",
            )

    if current_idx > 0:
        ax.scatter(
            coords[current_idx - 1, 0],
            coords[current_idx - 1, 1],
            c="red",
            s=150,
            marker="o",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"latent trajectory (PCA) | window: [{start_idx}, {current_idx})")

    margin = 0.05
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = (x_max - x_min) * margin + 1e-6
    y_pad = (y_max - y_min) * margin + 1e-6
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return img


def make_video(
    cfg,
    recording_name,
    t_start,
    t_end,
    fps=30,
    window_seconds=0.5,
    latent_subdir="latent",
    figsize=(12.8, 7.2),
    dpi=100,
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

    print(f"PCA 降维: {segment.shape[0]} 样本 x {segment.shape[1]} 维")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(segment)
    print(f"  解释方差比: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    duration = t_end - t_start
    n_frames = int(round(duration * fps))
    n_points = len(coords)

    latent_fs = n_points / duration if duration > 0 else 100.0
    window_size = max(2, int(round(window_seconds * latent_fs)))

    save_name = f"{recording_name}_pca_{t_start:.1f}_{t_end:.1f}.mp4"
    save_path = latent_dir / save_name

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)

    for f in range(n_frames):
        current_idx = min(n_points, int((f + 1) * n_points / n_frames))
        current_idx = max(1, current_idx)
        img = render_frame(coords, current_idx, window_size, figsize, dpi)
        writer.append_data(img)
        if (f + 1) % 30 == 0 or f == n_frames - 1:
            print(f"  渲染帧 {f + 1}/{n_frames}")

    writer.close()
    print(f"\n已保存视频: {save_path}")
    print(f"  时长: {duration:.1f}s, 帧数: {n_frames}, fps: {fps}")
    print(f"  latent 点数: {n_points}, 窗口时长: {window_seconds}s, 窗口点数: {window_size}")


def main():
    parser = argparse.ArgumentParser(description="latent PCA 滑动窗口轨迹视频")
    parser.add_argument("--recording", required=True, help="recording 名（不含 _latent.npz）")
    parser.add_argument("--start", type=float, required=True, help="起始时间戳（秒）")
    parser.add_argument("--end", type=float, required=True, help="结束时间戳（秒）")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument(
        "--window",
        type=float,
        default=0.5,
        help="滑动窗口时长（秒），默认 0.5s",
    )
    args = parser.parse_args()

    cfg = load_config()
    make_video(
        cfg,
        args.recording,
        args.start,
        args.end,
        fps=args.fps,
        window_seconds=args.window,
    )


if __name__ == "__main__":
    main()