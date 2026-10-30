from pathlib import Path
import argparse
import matplotlib as mpl
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection

from utils import load_config, output_dir

def render_state_frame(
    coords,
    states,
    current_idx,
    window_size,
    state_cmap,
    figsize,
    dpi,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    start_idx = max(0, current_idx - window_size)
    visible_coords = coords[start_idx:current_idx]
    visible_states = states[start_idx:current_idx]

    if len(visible_coords) > 1:
        rel = np.linspace(0, 1, len(visible_coords))

        # 连线
        segments = np.stack([visible_coords[:-1], visible_coords[1:]], axis=1)
        line_alphas = rel[1:] ** 2
        line_colors = np.array(state_cmap(visible_states[1:] % state_cmap.N))
        line_colors[:, 3] = 0.2 + 0.6 * line_alphas

        lc = LineCollection(segments, colors=line_colors, linewidths=2.0)
        ax.add_collection(lc)

        # 散点
        point_alphas = 0.2 + 0.7 * (rel ** 2)
        point_sizes = 15 + 50 * rel
        point_colors = np.array(state_cmap(visible_states % state_cmap.N))
        point_colors[:, 3] = point_alphas

        for i in range(len(visible_coords)):
            ax.scatter(
                visible_coords[i, 0],
                visible_coords[i, 1],
                c=point_colors[i].reshape(1, -1),
                s=point_sizes[i],
                edgecolors="none",
            )

    # 当前点
    if current_idx > 0:
        current_state = states[current_idx - 1]
        current_color = np.array(state_cmap(current_state % state_cmap.N)).reshape(1, -1)
        ax.scatter(
            coords[current_idx - 1, 0],
            coords[current_idx - 1, 1],
            c=current_color,
            s=250,
            marker="o",
            edgecolors="black",
            linewidths=2.5,
            zorder=5,
        )

    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_title(f"state trajectory | frame {current_idx}/{len(coords)}", fontsize=16)

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


def make_states_video(
    cfg,
    recording_name,
    t_start,
    t_end,
    fps=60,
    window_seconds=0.5,
    latent_subdir="latent",
    output_size=(1920, 1080),
    dpi=100,
    cmap="tab20",
    n_neighbors=15,
    min_dist=0.1,
):
    out = output_dir(cfg)
    latent_dir = out / latent_subdir

    latent_path = latent_dir / f"{recording_name}_latent.npz"
    states_path = latent_dir / f"{recording_name}_states.npz"

    if not latent_path.exists():
        raise FileNotFoundError(f"未找到 latent: {latent_path}")
    if not states_path.exists():
        raise FileNotFoundError(f"未找到状态标签，请先运行 fit_hdpslds.py: {states_path}")

    latent_data = np.load(latent_path)
    state_data = np.load(states_path)

    latent = latent_data["latent"]
    states = state_data["states"]
    time = latent_data["time"]

    # 时间截取
    mask = (time >= t_start) & (time <= t_end)
    if not mask.any():
        raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无数据")

    segment_latent = latent[mask]
    segment_states = states[mask]

    print(f"UMAP 降维: {segment_latent.shape[0]} 样本 x {segment_latent.shape[1]} 维")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    coords = reducer.fit_transform(segment_latent)

    duration = t_end - t_start
    n_frames = int(round(duration * fps))
    n_points = len(coords)

    # 估计 latent 采样率，计算窗口点数
    latent_fs = n_points / duration if duration > 0 else 100.0
    window_size = max(2, int(round(window_seconds * latent_fs)))

    state_cmap = mpl.colormaps.get_cmap(cmap)
    figsize = (output_size[0] / dpi, output_size[1] / dpi)

    # 准备输出视频
    save_name = f"{recording_name}_states_umap_{t_start:.1f}_{t_end:.1f}.mp4"
    save_path = latent_dir / save_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(save_path),
        fourcc,
        fps,
        output_size,
    )

    for f in range(n_frames):
        current_idx = min(n_points, int((f + 1) * n_points / n_frames))
        current_idx = max(1, current_idx)

        umap_rgba = render_state_frame(
            coords,
            segment_states,
            current_idx,
            window_size,
            state_cmap,
            figsize,
            dpi,
        )

        # RGBA -> RGB -> BGR
        umap_rgb = umap_rgba[:, :, :3]
        umap_bgr = cv2.cvtColor(umap_rgb, cv2.COLOR_RGB2BGR)
        writer.write(umap_bgr)

        if (f + 1) % 60 == 0 or f == n_frames - 1:
            print(f"  已渲染 {f + 1}/{n_frames} 帧")

    writer.release()
    print(f"\n已保存: {save_path}")
    print(f"  时长: {duration:.1f}s, 帧数: {n_frames}, fps: {fps}")
    print(f"  latent 点数: {n_points}, 窗口: {window_seconds}s ({window_size} 点)")


def main():
    parser = argparse.ArgumentParser(description="state 上色 UMAP 动态视频")
    parser.add_argument("--recording", required=True, help="recording 名")
    parser.add_argument("--start", type=float, required=True, help="起始时间戳（秒）")
    parser.add_argument("--end", type=float, required=True, help="结束时间戳（秒）")
    parser.add_argument("--fps", type=int, default=60, help="视频帧率")
    parser.add_argument("--window", type=float, default=0.5, help="滑动窗口时长（秒）")
    parser.add_argument("--cmap", type=str, default="tab20", help="state colormap")
    args = parser.parse_args()

    cfg = load_config()
    make_states_video(
        cfg,
        args.recording,
        args.start,
        args.end,
        fps=args.fps,
        window_seconds=args.window,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()