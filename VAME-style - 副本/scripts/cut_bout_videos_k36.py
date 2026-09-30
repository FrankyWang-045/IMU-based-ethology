# -*- coding: utf-8 -*-
"""VAME-Style k36 单层标签 -> data2.mp4 bout 裁剪 + 同文件夹信号图。

2026-09-28 v1。仅覆盖 3 条 t_diff 已实测 + 探针人工验证的 pilot session：
  rec_000 / rec_005 / rec_010（video_t = csv_t + t_diff，见 v3_recovery/docs/ALIGNMENT.md）。
其余 session 一律拒绝进入本脚本（防"假设 t_diff=0"的历史错误，见
docs/video_crop_pitfalls.txt）。

防御清单（全部来自 docs/video_crop_pitfalls.txt，2026-09-27 立字据）：
1. 时间基：video_t = t_npz + t_diff；t_npz 为合成轴但原点与当年实测轴一致（t[0]=120.00 s）。
2. 窗中心：latent 行 i <-> 30 窗 [i, i+29]，中心 i+15；bout 区间整体 +15 采样点。
3. 视频与信号图共用同一选段函数（同池、同种子、同数量）。
4. 全程 round，不用 int；帧区间半开 [f0, f1)。
5. 切后校验实际写出帧数 == 目标帧数，不符即删并记失败。
6. FPS 从容器读取并断言 |fps-29.97|<0.1；容器总帧数/时长交叉校验。
7. 切前比对视频时长 vs (t[-1]+t_diff)：视频必须完整覆盖 IMU 区间（视频更长允许）。
8. 输出目录版本化（videos_k36_v12/），逻辑变更后新建目录，绝不原地续跑。

用法：
  python cut_bout_videos_k36.py --states 0,1,2          # 指定类别
  python cut_bout_videos_k36.py                        # 全部 36 类
  python cut_bout_videos_k36.py --probe                # 探针：每 session 最尖锐 gyro 突发 +-7 s
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]           # VAME-Style/
WS = ROOT.parent                                      # IMU/
NPZ_DIR = WS / "VAME-IMU" / "data" / "ds25"
RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "videos_k36_v12"

BASE = "//hulab.lumiani.ai/Data/WSB/Computational_Ethology/Data"
# session -> (视频, 来源 CSV 名, t_diff[s])；只准用这 3 条（ALIGNMENT.md 2026-09-27 实测+探针验证）
SESSIONS = {
    "rec_000": (f"{BASE}/8.25_social_all_paired/A1_C1_C1-1c/data2.mp4",
                "A1_C1_A1-6d", 4.409501),
    "rec_005": (f"{BASE}/8.25_social_all_paired/A4_B4_A4-1c/data2.mp4",
                "A4_B4_B4-ab", 8.666189),
    "rec_010": (f"{BASE}/8.6_2mouse_social/Test1_vs_Tool1/data2.mp4",
                "Test1_vs_Tool1-13", -5.370704),
}
STAGING = ROOT / "cache" / "video_src"   # 本地暂存（48 GB 源视频的逐字节拷贝，
# 切完经用户确认后可删；存在即优先用本地，避免网络随机读拖慢 seek+解码）


def resolve_video(s):
    local = STAGING / f"{s}.mp4"
    return str(local) if local.exists() else SESSIONS[s][0]
NOMINAL_FPS = 29.97
CENTER_OFFSET = 15   # 30 窗窗中心补偿（采样点）
MIN_BOUT = 3
N_PER = 30
LONG_POOL = 2 * N_PER

ACC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # acc xyz
GYR_COLORS = ["#d62728", "#9467bd", "#8c564b"]   # gyro xyz（另外三色，不透明）
ACC_NAMES = ["acc_x", "acc_y", "acc_z"]
GYR_NAMES = ["gyro_x", "gyro_y", "gyro_z"]


def load_session(s):
    d = np.load(NPZ_DIR / f"{s}.npz")
    lab = np.load(RESULTS / s / "labels_k36_v12.npy")
    t = d["t"]
    assert len(t) - len(lab) == 29, f"{s}: 标签长度 {len(lab)} != len(t)-29"
    return t, d["raw6"].astype(np.float64), lab


def check_video(video, s, t, td):
    """防御 6/7：容器校验 + 覆盖性校验。返回 (fps, total_frames, duration)。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"{s}: 无法打开视频 {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if abs(fps - NOMINAL_FPS) > 0.1:
        raise RuntimeError(f"{s}: 容器 FPS={fps} 与名义 {NOMINAL_FPS} 不符")
    dur = nf / fps
    need = float(t[-1] + td)
    if dur < need:
        raise RuntimeError(f"{s}: 视频时长 {dur:.1f}s 不覆盖 IMU 终点 {need:.1f}s（切片段必缺帧）")
    return fps, nf, dur


def find_bouts(labels, state):
    is_s = labels == state
    d = np.diff(is_s.astype(np.int8), prepend=0, append=0)
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= MIN_BOUT]


def select_bouts(bouts_by_sess, n_per, seed):
    """防御 3：唯一的选段函数。视频与信号图都从返回结果取。"""
    cands = []
    for s, bl in bouts_by_sess.items():
        for b0, b1 in bl:
            cands.append((s, b0, b1))
    if len(cands) <= n_per:
        return sorted(cands)
    cands = sorted(cands, key=lambda c: -(c[2] - c[1]))[:LONG_POOL]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cands), size=n_per, replace=False)
    return [cands[i] for i in sorted(idx)]


def window_rows(b0, b1, n_t):
    """防御 2：bout [b0,b1) + 窗中心 15，钳到时间轴内。"""
    r0 = min(b0 + CENTER_OFFSET, n_t - 1)
    r1 = min(b1 - 1 + CENTER_OFFSET, n_t - 1)
    return r0, r1


def cut_clip(video, fps, f0, n_frames, out_path):
    """防御 4/5：半开区间 [f0, f0+n)，写后校验帧数；0 帧/缺帧时重开重试一次。"""
    for attempt in (1, 2):
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
        got = 0
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            vw.write(frame)
            got += 1
        vw.release()
        cap.release()
        if got == n_frames:
            return got, (w, h)
        out_path.unlink(missing_ok=True)
    return 0, (0, 0)


def plot_signal(raw6, t, r0, r1, out_png, title):
    """与视频严格同窗：绘制 [r0, r1] 的 dyn6，gyro 另外三色不透明。"""
    tt = t[r0:r1 + 1]
    seg = raw6[r0:r1 + 1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, cols, names, chans, ylab in (
        (axes[0], ACC_COLORS, ACC_NAMES, slice(0, 3), "acc"),
        (axes[1], GYR_COLORS, GYR_NAMES, slice(3, 6), "gyro"),
    ):
        for k, (c, nm) in enumerate(zip(cols, names)):
            ax.plot(tt, seg[:, chans][:, k], color=c, lw=0.8, label=nm)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_title(title, fontsize=10)
    axes[1].set_xlabel("t [s] (CSV time axis, video_t = t + t_diff)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run(out_root, states, n_per):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data, meta = {}, {}
    for s in SESSIONS:
        t, raw6, lab = load_session(s)
        video, src, td = resolve_video(s), SESSIONS[s][1], SESSIONS[s][2]
        fps, nf, dur = check_video(video, s, t, td)
        data[s] = (t, raw6, lab)
        meta[s] = dict(video=video, src=src, td=td, fps=fps, vf=nf, vdur=dur)
        print(f"[check] {s}: fps={fps:.3f} frames={nf} dur={dur:.1f}s "
              f"IMU_end+td={t[-1] + td:.1f}s OK", flush=True)

    manifest_path = out_root / "manifest.csv"
    new_manifest = not manifest_path.exists()
    mf = open(manifest_path, "a", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    if new_manifest:
        mw.writerow(["class", "session", "src_csv", "b0", "b1", "r0", "r1",
                     "f0", "n_frames_target", "n_frames_written", "t_start", "t_end",
                     "video_ok", "png"])

    for st in states:
        sel = select_bouts({s: find_bouts(d[2], st) for s, d in data.items()},
                           n_per, seed=int(st))
        if not sel:
            print(f"[class {st:02d}] 无 bout，跳过", flush=True)
            continue
        cdir = out_root / f"class_{st:02d}"
        cdir.mkdir(exist_ok=True)
        n_ok = 0
        for s, b0, b1 in sel:
            t, raw6, _ = data[s]
            video, src, td = meta[s]["video"], meta[s]["src"], meta[s]["td"]
            fps = meta[s]["fps"]
            r0, r1 = window_rows(b0, b1, len(t))
            f0 = int(round((t[r0] + td) * fps))
            f1 = int(round((t[r1] + td) * fps))           # 半开右端
            nf_target = max(1, f1 - f0 + 1)
            stem = f"{s}__{src}__{t[r0]:08.2f}s__{nf_target / fps:04.1f}s"
            mp4 = cdir / f"{stem}.mp4"
            png = cdir / f"{stem}.png"
            if mp4.exists() and png.exists():
                n_ok += 1
                continue
            got, wh = cut_clip(video, fps, f0, nf_target, mp4)
            ok = got == nf_target
            if not ok:
                mp4.unlink(missing_ok=True)
                print(f"  [FAIL] {stem}: 帧数 {got}/{nf_target}", flush=True)
            else:
                plot_signal(raw6, t, r0, r1, png,
                            f"class {st:02d} | {s} | {src} | bout[{b0},{b1}) "
                            f"win[{r0},{r1}] video_frames[{f0},{f0 + got}) {wh[0]}x{wh[1]}")
                mw.writerow([st, s, src, b0, b1, r0, r1, f0, nf_target, got,
                             f"{t[r0]:.3f}", f"{t[r1]:.3f}", int(ok), png.name])
                n_ok += 1
        mf.flush()
        print(f"[class {st:02d}] {n_ok}/{len(sel)} 段", flush=True)
    mf.close()
    print(f"完成 -> {out_root}", flush=True)


def run_probe(out_root):
    """每 session 取 gyro 能量最尖锐突发，切 ±7 s 探针（人工核对偏移）。"""
    pdir = Path(out_root) / "_probe"
    pdir.mkdir(parents=True, exist_ok=True)
    for s, (_, src, td) in SESSIONS.items():
        t, raw6, _ = load_session(s)
        video = resolve_video(s)
        fps, nf, dur = check_video(video, s, t, td)
        gyr_e = np.sum(raw6[:, 3:6] ** 2, axis=1)
        d = np.abs(np.diff(gyr_e, prepend=gyr_e[0]))
        margin = int(7 * (1.0 / np.median(np.diff(t))))   # ±7 s 的采样点数
        i = int(np.argmax(d[margin:-margin])) + margin
        r0 = max(0, i - margin)
        r1 = min(len(t) - 1, i + margin)
        f0 = int(round((t[r0] + td) * fps))
        nf_target = int(round((t[r1] + td) * fps)) - f0 + 1
        stem = f"probe__{s}__{src}__{t[i]:08.2f}s"
        mp4 = pdir / f"{stem}.mp4"
        got, wh = cut_clip(video, fps, f0, nf_target, mp4)
        ok = got == nf_target
        print(f"[probe] {s}: spike@t={t[i]:.2f}s -> {mp4.name} "
              f"frames {got}/{nf_target} {'OK' if ok else 'FAIL'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=None, help="逗号分隔类别（缺省全部）")
    ap.add_argument("--n-per", type=int, default=N_PER)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.probe:
        run_probe(args.out)
        return
    if args.states:
        states = sorted(int(x) for x in args.states.split(","))
    else:
        states = list(range(36))
    run(args.out, states, args.n_per)


if __name__ == "__main__":
    main()
