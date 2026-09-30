# -*- coding: utf-8 -*-
"""闪烁成因探针：聚类损失能救闪烁吗？

量化三件事（冻结模型 hmm_k36_v12，12 条口径）：
  A. 无 sticky 解码的闪烁强度：run 长度分布、每秒切换次数
  B. 似然平局度：后验 top1-top2 log 间隔分布；短 run（1-2 帧）的间隔
     是否远小于长 run —— 是 → 闪烁=发射高斯边界噪声，与 latent 几何无关
  C. latent 时间平滑度：相邻 mu 距离 vs 间隔 10 帧距离（窗重叠 29/30
     帧 → mu 本应极平滑；若真平滑则闪烁不可能源于 mu 跳变）
附带：拟合原貌转移矩阵 A 的对角统计。
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vamestyle.dataset import CFG, ROOT, sessions  # noqa: E402
from vamestyle.joint import K  # noqa: E402


def run_lengths(lab):
    d = np.diff(lab)
    starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
    ends = np.concatenate([starts[1:], [len(lab)]])
    return ends - starts


def main():
    rec = joblib.load(ROOT / "models" / f"hmm_k{K}_v12.joblib")
    model = rec["model"]
    A = model.transmat_
    diag = np.diag(A).copy()
    off = A.copy(); np.fill_diagonal(off, np.nan)
    print(f"[A] 对角 A_ii：均值 {np.nanmean(diag):.4f} 中位 {np.median(diag):.4f} "
          f"最小 {diag.min():.4f}")
    print(f"[A] 非对角最大值：{np.nanmax(off):.4f}（sticky κ=0.98 远高于它）")

    import copy
    sm = copy.deepcopy(model)
    As = sm.transmat_ * 0.02
    As[np.diag_indices_from(As)] += 0.98
    sm.transmat_ = As / As.sum(axis=1, keepdims=True)

    ex = set(CFG["data"]["exclude"])
    fs = CFG["data"]["fs"]
    plain_rl, sticky_rl = [], []
    margins_short, margins_long = [], []   # top1-top2 log 间隔
    for s in sessions():
        if s.name in ex:
            continue
        z = np.load(ROOT / "cache" / f"zfeat_{s.name}.npy")
        lp = np.nan  # 占位
        lab_p = model.predict(z)
        lab_s = sm.predict(z)
        plain_rl.append(run_lengths(lab_p))
        sticky_rl.append(run_lengths(lab_s))
        # 后验间隔（跨全部帧，predict_proba 前向-后向）
        post = model.predict_proba(z)
        top2 = np.sort(post, axis=1)[:, -2:]
        with np.errstate(divide="ignore"):
            m = np.log(top2[:, 1]) - np.log(top2[:, 0])
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        short = (run_lengths(lab_p) <= 2)
        # 与 m 对齐：重建逐帧 run 长度
        d = np.diff(lab_p)
        st = np.concatenate([[0], np.nonzero(d)[0] + 1])
        en = np.concatenate([st[1:], [len(lab_p)]])
        rl_t = np.repeat(en - st, en - st)
        margins_short.append(m[rl_t <= 2])
        margins_long.append(m[rl_t >= 25])
        print(f"[decode] {s.name}: 无sticky {len(plain_rl[-1])} runs, "
              f"sticky {len(sticky_rl[-1])} runs")

    p = np.concatenate(plain_rl); sk = np.concatenate(sticky_rl)
    print(f"\n[A] 无 sticky：runs={len(p)} 中位 {np.median(p):.0f} 帧 "
          f"({np.median(p) / fs:.2f}s) 1帧run占 {(p == 1).mean():.1%} "
          f"切换 {(len(p)) / (sum(p) / fs / 60):.0f} 次/分钟")
    print(f"[A] sticky  ：runs={len(sk)} 中位 {np.median(sk):.0f} 帧 "
          f"({np.median(sk) / fs:.2f}s) 1帧run占 {(sk == 1).mean():.1%} "
          f"切换 {len(sk) / (sum(sk) / fs / 60):.0f} 次/分钟")

    ms = np.concatenate(margins_short); ml = np.concatenate(margins_long)
    print(f"\n[B] top1-top2 log 间隔（自然对数，>0）：")
    print(f"  短run(≤2帧)帧：中位 {np.median(ms):.3f}，<0.5 占 {(ms < 0.5).mean():.1%}，"
          f"<0.1 占 {(ms < 0.1).mean():.1%}")
    print(f"  长run(≥25帧)帧：中位 {np.median(ml):.3f}，<0.5 占 {(ml < 0.5).mean():.1%}，"
          f"<0.1 占 {(ml < 0.1).mean():.1%}")

    # C. mu 时间平滑度
    mu = np.load(ROOT / "cache" / "mu_rec_000.npy").astype(np.float64)
    d1 = np.linalg.norm(np.diff(mu, axis=0), axis=1)
    d10 = np.linalg.norm(mu[10:] - mu[:-10], axis=1)
    print(f"\n[C] mu 平滑度 rec_000：相邻帧距离中位 {np.median(d1):.3f}，"
          f"间隔10帧 {np.median(d10):.3f}（比值 {np.median(d10) / np.median(d1):.1f}"
          f"，理想近 10；比值小 = latent 本身很平滑）")


if __name__ == "__main__":
    main()
