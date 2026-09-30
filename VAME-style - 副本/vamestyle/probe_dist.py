# -*- coding: utf-8 -*-
"""读 mixing_analysis.csv，打印类级 r 分布全景。"""
import csv
from collections import defaultdict

import numpy as np

rows = list(csv.reader(open(r"F:\Kimi\IMU\VAME-Style\results\mixing_analysis.csv", encoding="utf-8")))
hdr, data = rows[0], rows[1:]
byf = defaultdict(list)
for r in data:
    byf[int(r[0])].append(float(r[1]))
print(f"{'类':>4} {'n':>6} {'r50':>6} {'r75':>6} {'r90':>6} {'>0.6':>6} {'<0.3':>6}")
stat = []
for fid in sorted(byf):
    rs = np.array([x for x in byf[fid] if not np.isnan(x)])
    if len(rs) < 30:
        continue
    stat.append((fid, len(rs), *np.percentile(rs, [50, 75, 90]),
                 (rs > 0.6).mean(), (rs < 0.3).mean()))
stat.sort(key=lambda x: -x[3])
for fid, n, q50, q75, q90, hi, lo in stat:
    print(f"{fid:>4} {n:>6} {q50:>6.2f} {q75:>6.2f} {q90:>6.2f} "
          f"{hi:>6.0%} {lo:>6.0%}")
allr = np.array([float(r[1]) for r in data if r[1] != "nan"])
print(f"\n全 bout r 分位: 50={np.percentile(allr,50):.2f} "
      f"75={np.percentile(allr,75):.2f} 90={np.percentile(allr,90):.2f}")
