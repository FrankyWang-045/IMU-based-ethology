from pathlib import Path
import argparse

# ---------- 修复 sklearn 兼容性 ----------
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
# ----------------------------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA

from utils import load_config, output_dir


def render_pca_frame(coords, current_idx, window_size, figsize, dpi):
    """渲染单帧 PCA 轨迹图（滑动窗口 + 淡出）。"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    start_idx = max(0, current_idx - window_size)
    visible = coords[start_idx:current_idx]

    if len(visible) > 1:
        rel = np.linspace(0, 1, len(visible))
        segments = np.stack([visible[:-1], visible[1:]], axis=1)
        alphas = rel[1:] ** 2

        colors = np.zeros((len(segments), 4))
        colors[:, 0] = 0.27
        colors[:, 1] = 0.51
        colors[:, 2] = 0.71
        colors[:, 3] = 0.1 + 0.6 * alphas

        lc = LineCollection(segments, colors=colors, linewidths=2.0)
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
            )

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

    ax.set_xlabel("PC 1", fontsize=14)
    ax.set_ylabel("PC 2", fontsize=14)
    ax.set_title(f"latent trajectory (PCA) | t={current_idx}", fontsize=16)

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


def composite_video_pca(
    cfg,
    recording_name,
    video_path,
    t_start,
    t_end,
    fps=60,
    window_seconds=0.5,
    latent_subdir="latent",
    output_size=(1920, 1080),
    dpi=100,
):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"未找到视频: {video_path}")

    out = output_dir(cfg)
    latent_dir = out / latent_subdir
    npz_path = latent_dir / f"{recording_name}_latent.npz"

    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 latent: {npz_path}")

    # ---------- 读取视频信息 ----------
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(round(t_start * video_fps))
    end_frame = int(round(t_end * video_fps))
    n_frames = end_frame - start_frame

    print(f"视频信息: {video_fps:.2f} fps, 总帧数 {total_frames}")
    print(f"截取片段: 帧 {start_frame} ~ {end_frame} ({n_frames} 帧, {n_frames/video_fps:.2f}s)")

    # ---------- 加载并降维 latent ----------
    data = np.load(npz_path)
    latent = data["latent"]
    time = data["time"]

    mask = (time >= t_start) & (time <= t_end)
    if not mask.any():
        raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无 latent 数据")

    segment = latent[mask]

    print(f"PCA 降维: {segment.shape[0]} 样本 x {segment.shape[1]} 维")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(segment)
    print(f"  解释方差比: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"  累计解释方差: {pca.explained_variance_ratio_.sum():.3f}")

    duration = t_end - t_start
    latent_fs = len(segment) / duration if duration > 0 else 100.0
    window_size = max(2, int(round(window_seconds * latent_fs)))

    # PCA 图尺寸
    figsize = (output_size[0] / dpi, output_size[1] / dpi)

    # ---------- 准备输出视频 ----------
    save_name = f"{recording_name}_composite_pca_{t_start:.1f}_{t_end:.1f}.mp4"
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

        # 视频缩放到 1920x1080
        frame_resized = cv2.resize(frame, output_size)

        # 生成对应 PCA 帧
        latent_idx = min(len(coords), int((f + 1) * len(coords) / n_frames))
        latent_idx = max(1, latent_idx)
        pca_rgba = render_pca_frame(coords, latent_idx, window_size, figsize, dpi)

        # RGBA -> RGB -> BGR
        pca_rgb = pca_rgba[:, :, :3]
        pca_bgr = cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR)

        # 上下拼接
        composite = np.vstack([frame_resized, pca_bgr])

        writer.write(composite)

        if (f + 1) % 60 == 0 or f == n_frames - 1:
            print(f"  已处理 {f + 1}/{n_frames} 帧")

    cap.release()
    writer.release()
    print(f"\n已保存合成视频: {save_path}")
    print(f"  分辨率: {output_size[0]}x{output_size[1]*2}, 帧率: {fps} fps")


def main():
    parser = argparse.ArgumentParser(description="将 PCA 轨迹合成到现有视频")
    parser.add_argument("--recording", required=True, help="recording 名（不含 _latent.npz）")
    parser.add_argument("--video_path", required=True, help="原始视频完整路径")
    parser.add_argument("--start", type=float, required=True, help="片段起始时间（秒）")
    parser.add_argument("--end", type=float, required=True, help="片段结束时间（秒）")
    parser.add_argument("--window", type=float, default=0.5, help="PCA 滑动窗口时长（秒）")
    args = parser.parse_args()

    cfg = load_config()
    composite_video_pca(
        cfg,
        args.recording,
        args.video_path,
        args.start,
        args.end,
        window_seconds=args.window,
    )


if __name__ == "__main__":
    main()