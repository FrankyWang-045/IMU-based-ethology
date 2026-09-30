# -*- coding: utf-8 -*-
"""类内解剖：周期多数派 vs 少数派 bout 对比（读 mixing_analysis.csv）。

对每个 n>=50 的 flat 类：
  少数派 = 类内 r 最低 10%（疑似非周期）；多数派 = 类内 r 最高 50%。
对比：时长、logE、置信度、θ、是否嵌在长同类运行段内（前后邻同类）、
母状态分布。判定 H1（粘性吸收：低置信+嵌长段）vs H3（能量收编：
logE 与多数派相近）。
"""
import csv
from collections import Counter, defaultdict

import numpy as np

CSV = r"F:\Kimi\IMU\VAME-Style\results\mixing_analysis.csv"
rows = list(csv.reader(open(CSV, encoding="utf-8")))
hdr, data = rows[0], rows[1:]
byf = defaultdict(list)
for r in data:
    byf[int(r[0])].append(dict(
        flat=int(r[0]), r=float(r[1]), dur=float(r[2]), logE=float(r[3]),
        conf=float(r[4]), state=int(r[5]), theta=float(r[6]),
        prev=int(r[7]), next=int(r[8]), sess=r[9]))

print(f"{'类':>4} {'n':>5} | 少数派(r低10%) 时长/logE/置信/嵌段 | 多数派(r高50%) 时长/logE/置信")
tbl = []
for fid in sorted(byf):
    lst = [x for x in byf[fid] if not np.isnan(x["r"])]
    n = len(lst)
    if n < 50:
        continue
    rs = np.array([x["r"] for x in lst])
    lo_thr, hi_thr = np.percentile(rs, 10), np.median(rs)
    low = [x for x in lst if x["r"] <= lo_thr]
    hig = [x for x in lst if x["r"] >= hi_thr]
    def agg(grp):
        d = np.mean([x["dur"] for x in grp])
        e = np.mean([x["logE"] for x in grp])
        c = np.mean([x["conf"] for x in grp])
        emb = np.mean([x["prev"] == fid and x["next"] == fid for x in grp])
        return d, e, c, emb
    dl, el, cl, eml = agg(low)
    dh, eh, ch, emh = agg(hig)
    tbl.append((fid, n, dl, el, cl, eml, dh, eh, ch, emh))
    print(f"{fid:>4} {n:>5} | {dl:5.2f}s {el:5.2f} {cl:.3f} {eml:4.0%}   | "
          f"{dh:5.2f}s {eh:5.2f} {ch:.3f} {emh:4.0%}")

# 汇总：少数派 vs 多数派的差
L = np.array([[t[2], t[3], t[4], t[5]] for t in tbl])
H = np.array([[t[6], t[7], t[8], t[9]] for t in tbl])
print("\n[汇总] 少数派减多数派（>0 表示少数派更高）：")
for i, nm in enumerate(["时长", "logE", "置信", "嵌段率"]):
    d = L[:, i] - H[:, i]
    print(f"  {nm:6s}: 均值 {d.mean():+.3f}, 少数派更高的类占 {(d > 0).mean():.0%}")

# H1 细节：少数派嵌段结构——prev/next 同类占比、prev/next 类分布
print("\n[H1] 少数派 bout 的邻段去向（跨全部类汇总 top）：")
pc, nc = Counter(), Counter()
for fid in byf:
    lst = [x for x in byf[fid] if not np.isnan(x["r"])]
    if len(lst) < 50:
        continue
    lo_thr = np.percentile([x["r"] for x in lst], 10)
    for x in lst:
        if x["r"] <= lo_thr:
            pc[x["prev"]] += 1
            nc[x["next"]] += 1
print("  前邻:", pc.most_common(8))
print("  后邻:", nc.most_common(8))
