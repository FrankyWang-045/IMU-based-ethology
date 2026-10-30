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

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import umap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection

from utils import load_config, output_dir


def load_or_make_embedding(latent_dir, recording, latent, n_neighbors=15, min_dist=0.1):
    """加载或计算共享 UMAP 嵌入，确保与 island_outlines.py 坐标一致。"""
    p = Path(latent_dir) / f"{recording}_embedding.npy"
    if p.exists():
        print(f"使用缓存 embedding: {p}")
        return np.load(p)
    print("计算共享 embedding UMAP(15, 0.1, rs42) ...")
    emb = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    ).fit_transform(latent)
    np.save(p, emb)
    return emb


def render_umap_frame(
    coords,
    islands,
    extent,
    D,
    current_idx,
    window_size,
    figsize,
    dpi,
    cmap="tab20",
):
    """渲染单帧 UMAP：岛屿范围 + 游走轨迹（滑动窗口 + 淡出）。"""
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ---- 密度地形背景 ----
    if D is not None:
        ax.contourf(D, extent=extent, levels=15, cmap="terrain", alpha=0.35)

    # ---- 绘制岛屿范围 ----
    n_islands = len(islands)
    colors = [cmap(i % 20) for i in range(n_islands)]

    for i, isl in enumerate(islands):
        contour = np.asarray(isl["contour"])
        if len(contour) < 3:
            continue
        closed = np.vstack([contour, contour[0]])
        col = colors[i]
        # 填充
        ax.fill(contour[:, 0], contour[:, 1], color=col, alpha=0.25)
        # 边界
        ax.plot(closed[:, 0], closed[:, 1], color=col, lw=2.0)
        # 标注
        ax.annotate(
            str(isl["id"]),
            isl["peak"],
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            weight="bold",
            color=col,
        )

    # ---- 游走轨迹 ----
    start_idx = max(0, current_idx - window_size)
    visible = coords[start_idx:current_idx]

    if len(visible) > 1:
        rel = np.linspace(0, 1, len(visible))
        segments = np.stack([visible[:-1], visible[1:]], axis=1)
        alphas = rel[1:] ** 2

        line_colors = np.zeros((len(segments), 4))
        line_colors[:, 0] = 0.27  # steelblue R
        line_colors[:, 1] = 0.51
        line_colors[:, 2] = 0.71
        line_colors[:, 3] = 0.1 + 0.6 * alphas

        lc = LineCollection(segments, colors=line_colors, linewidths=2.0)
        ax.add_collection(lc)

        point_alphas = 0.15 + 0.75 * (rel ** 2)
        point_sizes = 15 + 45 * rel
        for i in range(len(visible)):
            ax.scatter(
                visible[i, 0],
                visible[i, 1],
                c="steelblue",
                s=point_sizes[i],
                alpha=point_alphas[i],
                edgecolors="none",
                zorder=4,
            )

    # 当前点
    if current_idx > 0:
        ax.scatter(
            coords[current_idx - 1, 0],
            coords[current_idx - 1, 1],
            c="red",
            s=200,
            marker="o",
            edgecolors="black",
            linewidths=2.0,
            zorder=5,
        )

    ax.set_xlabel("UMAP 1", fontsize=14)
    ax.set_ylabel("UMAP 2", fontsize=14)
    ax.set_title(f"latent trajectory + islands | t={current_idx}", fontsize=16)

    # 固定坐标范围与 islands npz 一致
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    plt.close(fig)
    return img


def composite_video_islands(
    cfg,
    recording_name,
    video_path,
    t_start,
    t_end,
    fps=60,
    window_seconds=0.5,
    latent_subdir="latent",
    islands_name=None,
    output_size=(1920, 1080),
    dpi=100,
    downsample=4,
    n_neighbors=15,
    min_dist=0.1,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"未找到视频: {video_path}")

    out = output_dir(cfg)
    latent_dir = out / latent_subdir
    latent_path = latent_dir / f"{recording_name}_latent.npz"

    if not latent_path.exists():
        raise FileNotFoundError(f"未找到 latent: {latent_path}")

    islands_name = islands_name or f"{recording_name}_islands.npz"
    islands_path = latent_dir / islands_name
    if not islands_path.exists():
        raise FileNotFoundError(f"未找到 islands: {islands_path}")

    # ---------- 读取视频信息 ----------
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(round(t_start * video_fps))
    end_frame = int(round(t_end * video_fps))
    n_frames = end_frame - start_frame

    print(f"视频信息: {video_fps:.2f} fps, 总帧数 {total_frames}")
    print(f"截取片段: 帧 {start_frame} ~ {end_frame} ({n_frames} 帧, {n_frames / video_fps:.2f}s)")

    # ---------- 加载 latent 与 islands ----------
    data = np.load(latent_path)
    latent_full = data["latent"].astype(np.float64)
    time_full = data["time"] if "time" in data else np.arange(len(latent_full))

    # 与 island_outlines.py 保持一致：对 latent 降采样后算 UMAP
    latent_ds = np.ascontiguousarray(latent_full[::downsample])
    time_ds = time_full[::downsample]

    emb = load_or_make_embedding(latent_dir, recording_name, latent_ds, n_neighbors, min_dist)

    mask = (time_ds >= t_start) & (time_ds <= t_end)
    if not mask.any():
        raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无 latent 数据")

    coords = emb[mask]
    segment_time = time_ds[mask]

    # 加载岛屿
    islands_data = np.load(islands_path, allow_pickle=True)
    extent = islands_data["extent"]
    D = islands_data["D"]
    raw_contours = islands_data["raw_contours"]
    island_ids = islands_data["island_ids"]

    islands = []
    for iid, contour in zip(island_ids, raw_contours):
        contour = np.asarray(contour)
        peak = contour.mean(axis=0)
        islands.append({"id": int(iid), "contour": contour, "peak": peak})

    print(f"加载 {len(islands)} 个岛屿, extent={extent}")

    # ---------- 计算窗口 ----------
    duration = t_end - t_start
    ds_fs = len(time_full) / (time_full[-1] - time_full[0]) / downsample if len(time_full) > 1 else 25.0
    window_size = max(2, int(round(window_seconds * ds_fs)))

    figsize = (output_size[0] / dpi, output_size[1] / dpi)

    save_name = f"{recording_name}_composite_islands_{t_start:.1f}_{t_end:.1f}.mp4"
    save_path = out / save_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(save_path),
        fourcc,
        fps,
        (output_size[0], output_size[1] * 2),
    )

    # ---------- 逐帧处理 ----------
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"警告: 第 {f} 帧读取失败，提前结束")
            break

        frame_resized = cv2.resize(frame, output_size)

        # 按视频真实时间对齐 latent 索引
        target_time = t_start + f / video_fps
        current_idx = int(np.searchsorted(segment_time, target_time))
        current_idx = min(len(coords) - 1, max(1, current_idx))

        umap_rgba = render_umap_frame(
            coords,
            islands,
            extent,
            D,
            current_idx,
            window_size,
            figsize,
            dpi,
        )

        umap_rgb = umap_rgba[:, :, :3]
        umap_bgr = cv2.cvtColor(umap_rgb, cv2.COLOR_RGB2BGR)

        composite = np.vstack([frame_resized, umap_bgr])
        writer.write(composite)

        if (f + 1) % 60 == 0 or f == n_frames - 1:
            print(f"  已处理 {f + 1}/{n_frames} 帧")

    cap.release()
    writer.release()
    print(f"\n已保存合成视频: {save_path}")
    print(f"  分辨率: {output_size[0]}x{output_size[1] * 2}, 帧率: {fps} fps")


def main():
    parser = argparse.ArgumentParser(description="将 UMAP 轨迹+岛屿范围合成到现有视频")
    parser.add_argument("--recording", required=True, help="recording 名（不含 _latent.npz）")
    parser.add_argument("--video_path", required=True, help="原始视频完整路径")
    parser.add_argument("--start", type=float, required=True, help="片段起始时间（秒）")
    parser.add_argument("--end", type=float, required=True, help="片段结束时间（秒）")
    parser.add_argument("--islands_name", type=str, default=None, help="岛屿 npz 文件名（默认 <recording>_islands.npz）")
    parser.add_argument("--downsample", type=int, default=4, help="必须与生成 islands 时一致（默认 4）")
    parser.add_argument("--fps", type=int, default=60, help="输出帧率")
    parser.add_argument("--window", type=float, default=0.5, help="UMAP 滑动窗口时长（秒）")
    args = parser.parse_args()

    cfg = load_config()
    composite_video_islands(
        cfg,
        args.recording,
        args.video_path,
        args.start,
        args.end,
        fps=args.fps,
        window_seconds=args.window,
        islands_name=args.islands_name,
        downsample=args.downsample,
    )


if __name__ == "__main__":
    main()