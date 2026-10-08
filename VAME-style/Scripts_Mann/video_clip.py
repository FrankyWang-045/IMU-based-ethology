"""根据单个 HSMM 结果文件，按 state 切割单个视频。

用法:
    python cut_video_by_hsmm.py --state 3 --seg 0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from utils import load_config, output_dir, hmm_hsmm_path

# ---------- 硬编码路径 ----------
VIDEO_PATH = Path(r"\\hulab.lumiani.ai\Data\WSB\Computational_Ethology\Data\8.25_social_all_paired\A1_C1_C1-1c\data2.mp4")   # 视频文件保持定死
REC_NAME = "A1_C1_C1-1c"                          # 手动设置录制名
OUT_DIR = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest\video_clips")

MIN_DURATION = 0.1
PADDING = 0
# --------------------------------


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


def cut_video(video_path, hsmm_path, out_dir, min_duration=0.1, padding=0, target_state=None, target_seg=None):
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
    print(f"FPS={fps}, 总帧数={total_frames}, HSMM 片段数={len(segments)}")

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
    HSMM_PATH = hmm_hsmm_path(cfg, REC_NAME)

    parser = argparse.ArgumentParser(description="按 HSMM state 切割视频")
    parser.add_argument("--state", type=int, default=None, help="只切割指定 state")
    parser.add_argument("--seg", type=int, default=None, help="只切割指定 state 的第 seg 个片段")
    args = parser.parse_args()

    cut_video(
        video_path=VIDEO_PATH,
        hsmm_path=HSMM_PATH,
        out_dir=OUT_DIR / REC_NAME,
        min_duration=MIN_DURATION,
        padding=PADDING,
        target_state=args.state,
        target_seg=args.seg,
    )


if __name__ == "__main__":
    main()