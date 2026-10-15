"""按最新 HSMM 结果自动切割视频。
只有 VIDEO_PATH 和 OUT_DIR 两个默认路径硬编码；
可通过 --video 和 --rec 覆盖。
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils import load_config, output_dir, latest_experiment_dir


# ---------- 默认路径（找不到 --video 时回退到这里）----------
VIDEO_PATH = Path(r"\\hulab.lumiani.ai\Data\WSB\Computational_Ethology\Data\8.25_social_all_paired\A3_C3_C3_1c\data2.mp4")
OUT_DIR = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\Batch3\videos")
# -----------------------------------------------------------


def find_segments(time, states):
    segments = []
    t = 0
    n = len(states)
    while t < n:
        s = t
        state = states[t]
        while t < n and states[t] == state:
            t += 1
        segments.append((state, time[s], time[t - 1]))
    return segments


def cut_video(
    video_path,
    hsmm_path,
    out_dir,
    min_duration=0.1,
    padding=0,
    target_state=None,
    target_seg=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hmm = np.load(hsmm_path)
    time = hmm["time"]
    states = hmm["states"]
    segments = find_segments(time, states)

    print(f"视频: {video_path}")
    print(f"HSMM: {hsmm_path}")
    print(f"FPS={fps:.2f}, 总帧数={total_frames}, 片段数={len(segments)}")

    seg_counter = {}
    cut_count = 0

    for state, start, end in segments:
        if target_state is not None and state != target_state:
            continue

        duration = end - start
        if duration < min_duration:
            continue

        seg_idx = seg_counter.get(state, 0)
        seg_counter[state] = seg_idx + 1

        if target_seg is not None and seg_idx != target_seg:
            continue

        start_frame = max(0, int((start - padding) * fps))
        end_frame = min(total_frames - 1, int((end + padding) * fps))

        state_dir = out_dir / f"state_{state:02d}"
        state_dir.mkdir(exist_ok=True)

        out_path = state_dir / f"seg_{seg_idx:04d}_{start:.3f}s_{end:.3f}s.mp4"

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cut_count += 1
        print(f"  保存: {out_path}")

    cap.release()
    print(f"完成: 共切割 {cut_count} 个片段，输出到 {out_dir}")


def main():
    cfg = load_config()
    out = output_dir(cfg)

    parser = argparse.ArgumentParser(description="按 HSMM state 切割视频")
    parser.add_argument("--rec", type=str, default=None,
                        help="指定 recording 名，如 A1_C1_C1-1c")
    parser.add_argument("--video", type=str, default=None,
                        help="指定视频路径；未指定则使用代码里的 VIDEO_PATH")
    parser.add_argument("--state", type=int, default=None,
                        help="只切割指定 state")
    parser.add_argument("--seg", type=int, default=None,
                        help="只切割指定 state 的第 seg 个片段")
    parser.add_argument("--min_duration", type=float, default=0.1,
                        help="最短片段时长（秒）")
    parser.add_argument("--padding", type=float, default=0,
                        help="片段前后扩展（秒）")
    args = parser.parse_args()

    exp_dir = latest_experiment_dir(cfg)
    print(f"使用实验目录: {exp_dir}")

    # 按 --rec 过滤，未指定则取第一个
    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))
    if args.rec:
        hsmm_files = [p for p in hsmm_files if p.stem.replace("_hsmm", "") == args.rec]

    if not hsmm_files:
        raise FileNotFoundError(
            f"{exp_dir} 中没有匹配的 *_hsmm.npz"
            + (f" (rec={args.rec})" if args.rec else "")
        )

    hsmm_path = hsmm_files[0]
    rec_name = hsmm_path.stem.replace("_hsmm", "")

    video_path = Path(args.video) if args.video else VIDEO_PATH

    cut_video(
        video_path=video_path,
        hsmm_path=hsmm_path,
        out_dir=OUT_DIR / rec_name,
        min_duration=args.min_duration,
        padding=args.padding,
        target_state=args.state,
        target_seg=args.seg,
    )


if __name__ == "__main__":
    main()