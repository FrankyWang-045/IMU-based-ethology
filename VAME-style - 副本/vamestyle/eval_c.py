# -*- coding: utf-8 -*-
"""路线 C 验收：新口径（_c）vs 旧口径（_v12）姿态纯度量化对比。

指标：
  1. 每状态姿态桶熵 H = -Σ p_b log p_b / log(4)（1=桶均匀=最混杂）；
     旧 bout 桶标签用已拟合的 buckets_k36_v12_v2_k4；新 bout 以同一批
     桶中心（δ 空间）最近邻分配——姿态坐标定义完全一致，对比才公平。
  2. bout 级 θ（旋转系 grav 均值倾角）的每状态散布 std。
  3. 结构 sanity：usage 最大类占比、K_eff、bout 数。
  4. BIC 冒烟检查（注意：28 维 vs 29 维 BIC 不可严格比较，仅看数量级）。
产物：results/eval_c.csv（每状态一行：新旧熵/θstd/usage）
"""
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vamestyle.dataset import CFG, ROOT, sessions  # noqa: E402
from vamestyle.posture import V0, _align_rotation  # noqa: E402
from vamestyle.states import MIN_BOUT  # noqa: E402

K = 36
KB = 4
C = CFG["vame"]["time_window"] // 2


def entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(KB))


def old_stats():
    """旧口径：posture_bouts csv（含桶序列对齐 buckets npy）。"""
    rows = list(csv.reader(open(
        ROOT / "cache" / "posture_bouts_k36_v12_v2.csv", encoding="utf-8")))[1:]
    buckets = np.load(ROOT / "cache" / "buckets_k36_v12_v2_k4.npy")
    assert len(rows) == len(buckets), (len(rows), len(buckets))
    state_bk, state_th = {}, {}
    for r, bk in zip(rows, buckets):
        st = int(r[3])
        state_bk.setdefault(st, np.zeros(KB, np.int64))[int(bk)] += 1
        state_th.setdefault(st, []).append(float(r[11]))     # theta 列
    return state_bk, state_th


def new_stats():
    """新口径：labels_k36_v12_c 游程 → 旋转 grav 均值 → 最近桶中心。"""
    centers = np.load(ROOT / "cache" / "buckets_k36_v12_v2_k4_centers.npy")
    ex = set(CFG["data"]["exclude"])
    state_bk, state_th = {}, {}
    for s in sessions():
        if s.name in ex:
            continue
        lab = np.load(ROOT / "results" / s.name / "labels_k36_v12_c.npy")
        grav = s.grav[C:C + len(lab)].astype(np.float64)
        R = _align_rotation(np.median(s.grav, axis=0))
        grav_r = grav @ R.T
        d = np.diff(lab)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [len(lab)]])
        for a, b in zip(starts, ends):
            if b - a < MIN_BOUT:
                continue
            st = int(lab[a])
            gm_r = grav_r[a:b].mean(0)
            bk = int(np.argmin(((centers - (gm_r - V0)) ** 2).sum(1)))
            state_bk.setdefault(st, np.zeros(KB, np.int64))[bk] += 1
            state_th.setdefault(st, []).append(float(np.degrees(
                np.arccos(np.clip(-gm_r[2] / max(np.linalg.norm(gm_r), 1e-12),
                                  -1, 1)))))
    return state_bk, state_th


def main():
    obk, oth = old_stats()
    nbk, nth = new_stats()
    # usage
    u_old = {int(r["state"]): float(r["pct"]) for r in csv.DictReader(open(
        ROOT / "results" / "usage_k36_v12.csv", encoding="utf-8"))}
    u_new = {int(r["state"]): float(r["pct"]) for r in csv.DictReader(open(
        ROOT / "results" / "usage_k36_v12_c.csv", encoding="utf-8"))}

    rows, H_old, H_new = [], [], []
    print(f"{'状态':>4} {'旧熵':>6} {'新熵':>6} {'Δ熵':>7} | {'旧θstd':>7} "
          f"{'新θstd':>7} | {'旧usage':>7} {'新usage':>7}")
    for st in sorted(set(obk) | set(nbk)):
        ho = entropy(obk[st]) if st in obk and obk[st].sum() >= 30 else np.nan
        hn = entropy(nbk[st]) if st in nbk and nbk[st].sum() >= 30 else np.nan
        to = float(np.std(oth[st])) if st in oth and len(oth[st]) >= 30 else np.nan
        tn = float(np.std(nth[st])) if st in nth and len(nth[st]) >= 30 else np.nan
        H_old.append(ho); H_new.append(hn)
        rows.append((st, ho, hn, to, tn, u_old.get(st), u_new.get(st)))
        if not np.isnan(ho) and not np.isnan(hn):
            print(f"{st:>4} {ho:>6.3f} {hn:>6.3f} {hn - ho:>+7.3f} | "
                  f"{to:>7.1f} {tn:>7.1f} | {u_old.get(st, 0):>6.1f}% "
                  f"{u_new.get(st, 0):>6.1f}%")
    Ho = np.nanmean(H_old); Hn = np.nanmean(H_new)
    print(f"\n[桶熵] 均值：旧 {Ho:.3f} → 新 {Hn:.3f}（Δ={Hn - Ho:+.3f}，"
          f"负=变纯）")
    dec = sum(1 for a, b in zip(H_old, H_new)
              if not np.isnan(a) and not np.isnan(b) and b < a - 0.01)
    inc = sum(1 for a, b in zip(H_old, H_new)
              if not np.isnan(a) and not np.isnan(b) and b > a + 0.01)
    n = sum(1 for a, b in zip(H_old, H_new)
            if not np.isnan(a) and not np.isnan(b))
    print(f"[桶熵] 变纯(Δ<-0.01) {dec}/{n}，变混(Δ>+0.01) {inc}/{n}")
    to = np.nanmean([r[3] for r in rows]); tn = np.nanmean([r[4] for r in rows])
    print(f"[θstd] 每状态均值：旧 {to:.1f}° → 新 {tn:.1f}°")

    print(f"\n[结构] 旧：最大类 {max(u_old.values()):.1f}% 新：最大类 "
          f"{max(u_new.values()):.1f}%（红线 <10%）")
    ko = len([v for v in u_old.values() if v >= 1.0])
    kn = len([v for v in u_new.values() if v >= 1.0])
    print(f"[结构] K_eff：旧 {ko} → 新 {kn}")
    b_old = sum(1 for _ in csv.reader(open(
        ROOT / "results" / "bouts_k36_v12.csv", encoding="utf-8"))) - 1
    b_new = sum(1 for _ in csv.reader(open(
        ROOT / "results" / "bouts_k36_v12_c.csv", encoding="utf-8"))) - 1
    print(f"[结构] bout 数：旧 {b_old} → 新 {b_new}")

    rec_c = joblib.load(ROOT / "models" / "hmm_k36_v12_c.joblib")
    rec_o = joblib.load(ROOT / "models" / "hmm_k36_v12.joblib")
    print(f"\n[BIC 冒烟] 旧(28维) {rec_o['bic']:.0f} vs 新(29维) "
          f"{rec_c['bic']:.0f}（维度不同，不可严格比较，仅看数量级；"
          f"新口径 conv={rec_c['converged']}）")

    with open(ROOT / "results" / "eval_c.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "H_old", "H_new", "theta_std_old",
                    "theta_std_new", "usage_old_pct", "usage_new_pct"])
        w.writerows(rows)
    print("-> results/eval_c.csv")


if __name__ == "__main__":
    main()
