# -*- coding: utf-8 -*-
"""帧级探针：类内低 r 帧的位置与成因（H1/H2/插值伪影）。

对每个 session：
  r_tr = raw acc 3 轴 periodicity_trace 取 max（与 mixing v2 同口径）
  低 r 帧 = 全局 r_tr 最低 5%
检查：
  (a) 在 bout 内的相对位置（0=起点，1=终点）→ 居中=粘性吸收嫌疑
  (b) interp_mask 占比 vs 全局 → 插值伪影嫌疑
  (c) 幅度（linacc RMS）vs 全体 → 低 r 是否只是低幅度
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vamestyle.dataset import CFG, ROOT, sessions  # noqa: E402
from vamestyle.joint import K, KB, build_frame_joint, prune  # noqa: E402
from vamestyle.mixing import periodicity_trace  # noqa: E402


def main():
    frame_joint, _, usage, _, _ = build_frame_joint()
    remap, _ = prune(usage)
    ex = set(CFG["data"]["exclude"])
    C = CFG["vame"]["time_window"] // 2

    pos_all, interp_low, n_low, n_all = [], 0.0, 0, 0
    amp_low, amp_all = [], []
    for s in sessions():
        if s.name in ex:
            continue
        jj = frame_joint[s.name]
        f = remap[jj]
        T = len(f)
        raw = s.raw6[C:C + T, 0:3].astype(np.float64)
        r_tr = np.nanmax(np.stack([periodicity_trace(raw[:, i])
                                   for i in range(3)]), axis=0)
        lin = s.feat9[C:C + T, 0:3]
        amp = np.sqrt(np.mean(lin * lin, axis=1))
        im = s.interp_mask[C:C + T] if s.interp_mask is not None \
            else np.zeros(T, np.uint8)
        ok = ~np.isnan(r_tr)
        thr = np.nanpercentile(r_tr, 5)
        low = ok & (r_tr <= thr)
        # bout 内相对位置
        d = np.diff(f)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [T]])
        si = np.repeat(np.arange(len(starts)), ends - starts)
        rel = (np.arange(T) - starts[si]) / np.maximum(ends[si] - starts[si], 1)
        pos_all.append(rel[low])
        interp_low += im[low].sum()
        n_low += low.sum()
        n_all += ok.sum()
        amp_low.append(amp[low])
        amp_all.append(amp[ok])
    pos = np.concatenate(pos_all)
    a_low = np.concatenate(amp_low)
    a_all = np.concatenate(amp_all)
    print(f"低 r 帧共 {n_low} / {n_all}")
    print(f"(a) bout 内相对位置: 均值={pos.mean():.2f} 中位={np.median(pos):.2f} "
          f"|<0.2|={np.mean(pos < 0.2):.1%} 0.2-0.8={np.mean((pos >= 0.2) & (pos <= 0.8)):.1%} >0.8={np.mean(pos > 0.8):.1%}")
    print(f"    （若粘性居中吸收，应集中于 0.5 附近；若边界切换伪影，应集中于两端）")
    print(f"(b) interp_mask 占比: 低r帧 {interp_low / n_low:.2%} vs 全体 "
          f"{'?' if n_all == 0 else ''}")
    tot_interp = 0
    for s in sessions():
        if s.name in ex:
            continue
        jj = frame_joint[s.name]
        T = len(jj)
        if s.interp_mask is not None:
            tot_interp += s.interp_mask[C:C + T].sum()
    print(f"    全体插值帧占比 {tot_interp / n_all:.2%}")
    print(f"(c) 幅度 g: 低r帧中位 {np.median(a_low):.3f} vs 全体中位 {np.median(a_all):.3f}")


if __name__ == "__main__":
    main()
