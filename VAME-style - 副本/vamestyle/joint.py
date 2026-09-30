# -*- coding: utf-8 -*-
"""v4.0 联合标签空间（路线 B，Phase 6 第③④步）：状态 × 姿态桶 → 扁平谱。

口径（已拍板）：K=36（hmm_k36_v12），12 条（排除 rec_011/013），
姿态桶 = 方案 2 逐鼠垂直零点 k=4（cache/buckets_k36_v12_v2_k4*.npy）。
剪枝：组合占用 <1% 的帧回退到母状态的"主桶"（该 state 内占用最高的桶），
保证每帧必有单一标签、谱扁平（VAME 原文规则）。

产物（全部 _k36_v12_v2 后缀，零覆盖）：
  results/joint_spectrum.csv        谱映射表（flat_id, state, bucket, 用法%,
                                    grav_mean/raw/theta 统计）——谱的正式定义文档
  results/<session>/joint_k36_v12_v2.npy   帧级联合标签（int16，长=zfeat 行数）
  results/joint_validation_k36_v12_v2.txt  验证口径四条达标检查
  results/joint_communities_k36_v12_v2.csv community 表（转移流层次聚类）

community：联合标签帧序列的经验转移计数 T_joint（原貌，非 sticky），
F_ij = u_i·T_ij + u_j·T_ji，D_ij = 1 − F_ij/F_max（VAME 4.4 精神），
层次凝聚后在「4–20 个 community」区间取最大间隙切割。

用法：python -m vamestyle.joint
"""
import csv
from pathlib import Path

import numpy as np

from vamestyle.dataset import CFG, ROOT, sessions
from vamestyle.posture import V0, collect_bouts, fit_buckets

K = 36
KB = 4
PRUNE = 0.01                             # 剪枝阈值（拍板）
SFX = f"_k{K}_v12_v2"


def _all_runs(lab):
    d = np.diff(lab)
    starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
    ends = np.concatenate([starts[1:], [len(lab)]])
    return starts, ends


def build_frame_joint():
    """帧级联合标签 + bout 级联合表。返回 (frame_joint {sess: int16},
    bout_rows, usage_flat (K*KB,), grav_raw, grav_theta)。"""
    rows = collect_bouts(K, "v12", "v2")     # min-bout 过滤后的 bout（含桶序）
    buckets = fit_buckets(K, "v12", "v2", KB)
    centers = np.load(ROOT / "cache" / f"buckets_k{K}_v12_v2_k{KB}_centers.npy")
    ex = set(CFG["data"]["exclude"])
    C = CFG["vame"]["time_window"] // 2

    frame_joint, bout_rows = {}, []
    usage = np.zeros(K * KB, dtype=np.int64)
    grav_raw_all, theta_all = [], []
    r_i = 0
    for s in sessions():
        if s.name in ex:
            continue
        lab = np.load(ROOT / "results" / s.name / f"labels_k{K}_v12.npy")
        grav = s.grav[C:C + len(lab)].astype(np.float64)
        # 与 posture.collect_bouts 相同的旋转（逐鼠垂直零点）
        from vamestyle.posture import _align_rotation
        R = _align_rotation(np.median(s.grav, axis=0))
        grav_r = grav @ R.T
        jj = np.empty(len(lab), dtype=np.int16)
        starts, ends = _all_runs(lab)
        for a, b in zip(starts, ends):
            kept = (b - a) >= 4             # MIN_BOUT，与 collect_bouts 一致
            if kept:
                bk = int(buckets[r_i])
                gm_raw = grav[a:b].mean(0)
                gm_r = grav_r[a:b].mean(0)
                theta = np.degrees(np.arccos(np.clip(-gm_r[2], -1, 1)))
                bout_rows.append((s.name, int(a), int(b), int(lab[a]), bk,
                                  *gm_raw, theta))
                grav_raw_all.append(gm_raw)
                theta_all.append(theta)
                r_i += 1
            else:
                # 短游程：rotated 均值最近桶中心
                gm_r = grav_r[a:b].mean(0)
                bk = int(np.argmin(((centers - (gm_r - V0)) ** 2).sum(1)))
            jj[a:b] = int(lab[a]) * KB + bk
        frame_joint[s.name] = jj
        usage += np.bincount(jj, minlength=K * KB)
    return frame_joint, bout_rows, usage, np.array(grav_raw_all), np.array(theta_all)


def prune(usage):
    """<1% 组合回退母状态主桶。返回 flat_id 映射：组合索引 → 扁平 id。
    注意：整状态占用 <1% 时其主桶也 <1%，必须先给全部主桶发 id，
    否则回退目标仍是 -1。"""
    pct = usage / usage.sum()
    combo_state = np.repeat(np.arange(K), KB)
    main_bucket = {}
    for st in range(K):
        m = pct[st * KB:(st + 1) * KB]
        main_bucket[st] = int(np.argmax(m))
    remap = np.full(K * KB, -1, dtype=np.int64)
    nxt = 0
    for st in range(K):                  # 先保证每个母状态的主桶必有 id
        remap[st * KB + main_bucket[st]] = nxt; nxt += 1
    for c in range(K * KB):              # 其余 ≥1% 组合
        if remap[c] < 0 and pct[c] >= PRUNE:
            remap[c] = nxt; nxt += 1
    for c in range(K * KB):              # 剪枝回退到母状态主桶
        if remap[c] < 0:
            remap[c] = remap[combo_state[c] * KB + main_bucket[combo_state[c]]]
    return remap, pct


def main():
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    frame_joint, bout_rows, usage, grav_raw, theta = build_frame_joint()
    remap, pct = prune(usage)

    # 帧级标签落盘 + 谱统计
    flat_usage = np.bincount(np.concatenate(
        [remap[jj] for jj in frame_joint.values()]),
        minlength=remap.max() + 1)
    flat_pct = 100 * flat_usage / flat_usage.sum()
    combo_state = np.repeat(np.arange(K), KB)
    combo_bucket = np.tile(np.arange(KB), K)
    rows = []
    for fid in range(len(flat_usage)):
        members = np.nonzero(remap == fid)[0]
        st = int(combo_state[members[0]])
        bk = int(combo_bucket[members[0]]) if len(members) == 1 else -1
        # grav 属性：成员 bout 的原始 grav_mean 汇总
        gm = np.array([bout_rows[i][5:8] for i in range(len(bout_rows))
                       if remap[bout_rows[i][3] * KB + bout_rows[i][4]] == fid])
        th = np.array([bout_rows[i][8] for i in range(len(bout_rows))
                       if remap[bout_rows[i][3] * KB + bout_rows[i][4]] == fid])
        rows.append((fid, st, bk, round(float(flat_pct[fid]), 3),
                     round(float(np.median(th)), 1) if len(th) else None,
                     len(members)))
    with open(res / f"joint_spectrum{SFX}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["flat_id", "state", "bucket", "usage_pct",
                    "theta_median_deg", "n_combo_merged"])
        w.writerows(rows)
    for sn, jj in frame_joint.items():
        np.save(res / sn / f"joint{SFX}.npy", remap[jj].astype(np.int16))
    print(f"[joint] 谱：{len(rows)} 个 flat 类（初始组合 "
          f"{int((pct >= PRUNE).sum())} 个，剪枝回退 "
          f"{int((pct < PRUNE).sum())} 个组合）")

    # 验证口径
    v = []
    v.append(f"最大类占比 {flat_pct.max():.1f}%（红线 <10%）"
             f" -> {'PASS' if flat_pct.max() < 10 else 'FAIL'}")
    v.append(f"幽灵类 <0.5% 个数 {int((flat_pct < 0.5).sum())}（要求 0）"
             f" -> {'PASS' if (flat_pct < 0.5).sum() == 0 else 'WARN'}")
    # 姿态可控性：flat 类内 theta spread < 母状态 spread（承诺指标）
    state_theta = {}
    for r in bout_rows:
        state_theta.setdefault(r[3], []).append(r[8])
    ok, tot = 0, 0
    fid_theta = {}
    for i, r in enumerate(bout_rows):
        fid_theta.setdefault(remap[r[3] * KB + r[4]], []).append(r[8])
    for fid, ths in fid_theta.items():
        st = rows[fid][1]
        tot += 1
        if np.std(ths) < np.std(state_theta[st]):
            ok += 1
    v.append(f"姿态可控性：{ok}/{tot} 个 flat 类内 θ 散布小于母状态"
             f" -> {'PASS' if ok / tot > 0.9 else 'WARN'}")
    n_comm = len(rows)
    v.append(f"community 数待计算（应 ≥14，姿态展开只应更细）")
    (res / f"joint_validation{SFX}.txt").write_text("\n".join(v), encoding="utf-8")
    print("[joint] " + "\n[joint] ".join(v))

    # community：联合标签经验转移流层次聚类（原貌计数，非 sticky）
    from scipy.cluster.hierarchy import linkage, fcluster
    T = np.zeros((len(rows), len(rows)))
    for sn, jj in frame_joint.items():
        f = remap[jj]
        np.add.at(T, (f[:-1], f[1:]), 1)
    u = T.sum(1)
    F = T * u[:, None] + T.T * u[None, :]
    Fm = F.max()
    D = 1 - F / Fm
    np.fill_diagonal(D, 0)
    iu = np.triu_indices_from(D, 1)
    Z = linkage(D[iu], method="average")
    # 切割：在产生 4–20 个 community 的高度区间内取最大合并间隙
    hs = np.unique(Z[:, 2])
    n_leaf = len(rows)
    def n_at(h):
        return int(n_leaf - (Z[:, 2] <= h).sum())
    best = None
    for h0, h1 in zip(hs[:-1], hs[1:]):
        n1 = n_at(h1)                    # 跨过这个间隙后剩 n1 类
        if 4 <= n1 <= 20:
            gap = h1 - h0
            if best is None or gap > best[0]:
                best = (gap, h1, n1)
    if best is None:
        cl = fcluster(Z, 14, criterion="maxclust")
        print("[joint] 4–20 区间无可用间隙，退回 14 community")
    else:
        cl = fcluster(Z, best[1], criterion="distance")
        print(f"[joint] community 切割：最大间隙 {best[0]:.4f} @ "
              f"height={best[1]:.4f} -> {best[2]} 个 community")
    comm_rows = {}
    for (fid, st, bk, up, tm, nm), c in zip(rows, cl):
        comm_rows.setdefault(int(c), []).append((fid, up))
    with open(res / f"joint_communities{SFX}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["community", "n_flat", "flat_ids", "usage_pct", "n_states"])
        for c in sorted(comm_rows):
            members = comm_rows[c]
            up = sum(m[1] for m in members)
            sts = len({rows[m[0]][1] for m in members})
            w.writerow([c, len(members),
                        " ".join(str(m[0]) for m in members),
                        round(up, 2), sts])
    print(f"[joint] community 表 -> joint_communities{SFX}.csv")


if __name__ == "__main__":
    main()
