# -*- coding: utf-8 -*-
"""v4.0  bout 视频裁剪 + 同目录信号图（2026-09-28）。

方案 A：仅用 3 条有实测 t_diff 的 pilot session（rec_000/005/010），
按 36 类单层 sticky-HMM 标签（labels_k36_v12）每类选 30 个 bout，
从各自 data2.mp4 裁剪，并把 bout 信号图（dyn6）存到同一文件夹。

对齐约定（依据 v3_recovery/docs/ALIGNMENT.md，2026-09-27 实测 + 探针验证）：
  video_t = csv_t + t_diff  （csv_t = npz['t']，v2.0 合成秒轴；ds25 与 100Hz 原点同为 120.00s）
  latent 标签行 i ↔ 30 帧窗（stride 1），bout [b0,b1) 整体 +15 帧补到窗中心
  FPS 以容器读取为准（应≈29.97），round 取帧，区间半开
选段规则（与 sanity 图/历史 v2 裁剪一致）：全部 bout（≥3 帧）→ 按长度降序取前 60
  → rng(类号) 无放回取 30；视频与信号图由同一次选段结果生成，天然一致。
防御（依据 docs/video_crop_pitfalls.txt）：输出目录版本化、切后校验实际帧数==目标、
  容器 FPS/总帧数/时长三重校验、超出视频末端的 bout 记录并跳过、manifest.csv 全量留痕。
产出：videos_k36_v12/class_XX/{sess}__{csv}__{t0:08.2f}s__{dur}s.mp4 与同名 .png
用法：python cut_bout_videos_v2.py [--states 0,1,2] [--out 目录]（幂等，已存在跳过）
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT.parent / "VAME-IMU" / "data" / "ds25"
BASE = "//hulab.lumiani.ai/Data/WSB/Computational_Ethology/Data"
# session -> (视频, 来源 CSV 名, t_diff[s])；video_t = csv_t + t_diff（实测，勿改）
VIDEO_MAP = {
    "rec_000": (f"{BASE}/8.25_social_all_paired/A1_C1_C1-1c/data2.mp4",
                "A1_C1_A1-6d", 4.409501),
    "rec_005": (f"{BASE}/8.25_social_all_paired/A4_B4_A4-1c/data2.mp4",
                "A4_B4_B4-ab", 8.666189),
    "rec_010": (f"{BASE}/8.6_2mouse_social/Test1_vs_Tool1/data2.mp4",
                "Test1_vs_Tool1-13", -5.370704),
}
CENTER_OFFSET = 15        # latent 30 帧窗的窗中心补偿（帧）
MIN_BOUT = 3              # 最短 bout（标签行）
N_PER = 30                # 每类目标段数
TOP_M = 60                # 长 bout 优先池
K = 36                    # 类别数（仅用于校验）
FS = 25.0                 # npz 采样率（仅用于信号图上下文）

# 信号图配色：acc 三轴 / gyro 三轴各自一组、全部不透明
ACC_C = ["#1f77b4", "#ff7f0e", "#2ca02c"]
GYR_C = ["#d62728", "#9467bd", "#17becf"]
AX_LBL = ["x", "y", "z"]


def load_inputs():
    t_arr, lab_arr, sig = {}, {}, {}
    for s in VIDEO_MAP:
        d = np.load(DATA_DIR / f"{s}.npz")
        t_arr[s] = d["t"]
        sig[s] = d["raw6"]                       # (n,6) acc(g) x3 + gyro(dps) x3
        lab = np.load(ROOT / "results" / s / "labels_k36_v12.npy")
        assert len(lab) == len(t_arr[s]) - 29, f"{s}: 标签长度 {len(lab)} != n-29"
        lab_arr[s] = lab
    return t_arr, lab_arr, sig


def check_video(s, t, td):
    """容器校验：FPS≈29.97、时长×fps≈总帧数、覆盖 npz 全程。返回 (fps, n_frames)。"""
    path, _, _ = VIDEO_MAP[s]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"{s}: 无法打开 {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = nfr / fps
    ok = abs(fps - 29.97) < 0.05
    print(f"[check] {s}: fps={fps:.3f} frames={nfr} dur={dur:.1f}s "
          f"npz 末端对应帧={round((t[-1] + td) * fps)} "
          f"{'OK' if ok and (t[-1] + td) < dur else '⚠ 需人工确认'}", flush=True)
    if not ok:
        raise RuntimeError(f"{s}: 容器 FPS={fps} 与 29.97 不符，停止（勿盲目切）")
    return fps, nfr


def find_bouts(labels, state):
    is_s = labels == state
    d = np.diff(is_s.astype(np.int8), prepend=0, append=0)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= MIN_BOUT]


def select_bouts(lab_arr, state, n_per=N_PER):
    cands = [(s, b0, b1) for s in lab_arr for b0, b1 in find_bouts(lab_arr[s], state)]
    if len(cands) > n_per:
        cands = sorted(cands, key=lambda c: -(c[2] - c[1]))[:TOP_M]
        rng = np.random.default_rng(int(state))
        idx = rng.choice(len(cands), size=min(n_per, len(cands)), replace=False)
        cands = [cands[i] for i in sorted(idx)]
    return cands


def bout_frames(t, td, fps, b0, b1, n_frames_cap):
    """半开区间取帧：起点窗中心对齐，末端=最后一个标签行+15 帧。"""
    r0 = min(b0 + CENTER_OFFSET, len(t) - 1)
    r1 = min(b1 - 1 + CENTER_OFFSET, len(t) - 1)
    f0 = int(round((t[r0] + td) * fps))
    f1 = int(round((t[r1] + td) * fps))
    nf = max(1, f1 - f0 + 1)
    over = f0 + nf - n_frames_cap          # 超出视频末端的帧数
    return f0, nf, over, t[r0]


def cut_clip(video, f0, n_frames, out_path):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         VIDEO_FPS[0], (w, h))
    got = 0
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame)
        got += 1
    vw.release()
    cap.release()
    return got


def plot_signal(sig_s, t, b0, b1, st, sess, out_path):
    """dyn6 信号图：acc(g)/gyro(dps) 两行，绝对 CSV 时间轴，bout 区橙色底纹。"""
    n = len(t)
    center = (b0 + b1) / 2
    half = max((b1 - b0) / 2 + 5 * FS, 6 * FS)
    lo, hi = int(max(0, center - half)), int(min(n - 1, center + half))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    for row, (cols, colors, name, unit) in enumerate([
            (range(0, 3), ACC_C, "acc", "g"),
            (range(3, 6), GYR_C, "gyro", "dps")]):
        ax = axes[row]
        for k, c in zip(cols, colors):
            ax.plot(t[lo:hi], sig_s[lo:hi, k], color=c, lw=0.8, label=f"{name}{AX_LBL[k % 3]}")
        ax.axvspan(t[b0], t[min(b1 - 1, n - 1)], color="orange", alpha=0.25, lw=0)
        ax.set_ylabel(f"{name} ({unit})")
        ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.6)
        ax.grid(alpha=0.3)
    axes[0].set_title(f"class {st:02d} | {sess} @ {t[b0]:.2f}s  "
                      f"bout {t[b0]:.2f}–{t[b1 - 1]:.2f}s ({(b1 - b0) / FS:.1f}s)")
    axes[1].set_xlabel("CSV time (s)")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


VIDEO_FPS = [29.97]       # 容器实测后回填，cut 用它保证帧间隔一致


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "videos_k36_v12"))
    ap.add_argument("--states", default=None, help="逗号分隔类 id 子集（探针用）")
    ap.add_argument("--replot", action="store_true",
                    help="只强制重画 PNG（统一格式），已存在的视频不重切")
    args = ap.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    t_arr, lab_arr, sig = load_inputs()
    # 容器校验并回填实测 FPS
    fps_tab = {}
    for s in VIDEO_MAP:
        fps, nfr = check_video(s, t_arr[s], VIDEO_MAP[s][2])
        fps_tab[s] = (fps, nfr)
    VIDEO_FPS[0] = fps_tab["rec_000"][0]

    if args.states:
        states = sorted(int(x) for x in args.states.split(","))
    else:
        states = list(range(K))
        for s, lab in lab_arr.items():
            assert lab.min() >= 0 and lab.max() < K, f"{s}: 标签越界 [0,{K})"

    manifest = out_root / "manifest.csv"
    new_file = not manifest.exists()
    mf = open(manifest, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(mf)
    if new_file:
        w.writerow(["class", "session", "b0", "b1", "t_start_csv", "dur_csv_s",
                    "f0", "n_target", "n_written", "over_end", "video", "png", "status"])

    total_new, total_skip, total_bad = 0, 0, 0
    for st in states:
        cands = select_bouts(lab_arr, st)
        if not cands:
            print(f"[class {st:02d}] 无 bout，跳过", flush=True)
            continue
        out_dir = out_root / f"class_{st:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        n_ok, n_bad = 0, 0
        for s, b0, b1 in cands:
            video, csv_name, td = VIDEO_MAP[s]
            t = t_arr[s]
            fps, nfr = fps_tab[s]
            f0, nf, over, t0 = bout_frames(t, td, fps, b0, b1, nfr)
            stem = f"{s}__{csv_name}__{t0:08.2f}s__{nf / fps:04.1f}s"
            vpath, ppath = out_dir / f"{stem}.mp4", out_dir / f"{stem}.png"
            status = "ok"
            if over > 0:
                status = f"over_end_{over}f"
            cached = vpath.exists() and ppath.exists() and not args.replot
            if cached:
                total_skip += 1
                n_ok += 1
                w.writerow([st, s, b0, b1, f"{t0:.3f}", f"{(b1 - b0) / FS:.3f}",
                            f0, nf, nf, over, vpath.name, ppath.name, "cached"])
                continue
            if args.replot and vpath.exists():
                got = nf                      # 视频已存在且校验过，只重画 PNG
            else:
                got = -1 if over > 0 else cut_clip(video, f0, nf, vpath)
            if got == nf:
                plot_signal(sig[s], t, b0, b1, st, s, ppath)
                n_ok += 1
                total_new += 1
            else:
                n_bad += 1
                total_bad += 1
                if vpath.exists():
                    vpath.unlink()
                status = f"bad_frames_{got}"
            w.writerow([st, s, b0, b1, f"{t0:.3f}", f"{(b1 - b0) / FS:.3f}",
                        f0, nf, got if got >= 0 else "", over,
                        vpath.name if got == nf else "", ppath.name if got == nf else "",
                        status])
            mf.flush()
        print(f"[class {st:02d}] {n_ok}/{len(cands)} 段"
              + (f"（失败 {n_bad}）" if n_bad else ""), flush=True)
    mf.close()
    print(f"完成：新建 {total_new}，缓存命中 {total_skip}，失败 {total_bad} → {out_root}",
          flush=True)


if __name__ == "__main__":
    main()
