# -*- coding: utf-8 -*-
"""路线 D 姿态审计：姿态信息被编码了多少、被识别了多少。

姿态真值：ds50 grav3 → 逐 session 垂直零点系（posture._align_rotation）→
  连续量 (θ_rad, sinφ, cosφ)；离散量：k-means 姿态类（k=4/8，MiniBatch）。
指标：
  编码：ridge 回归 mu→姿态真值，50/50 分割 held-out R²（逐维+均值）；
  识别：NMI(HMM状态, 姿态类)（adjusted，剔除偶然）、类内姿态纯度（weighted）；
对照：现行版 dyn6-VAE（25 Hz cache/mu + labels_k36_v12）同口径计算。
产物：runs/routeD_posture_vae_50hz/results/posture_audit.json
用法：venv_python scripts/routeD_posture_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(r"F:\Kimi\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
VAME_IMU = WORKSPACE / "VAME-IMU"
for p in (str(VAME_STYLE), str(VAME_IMU)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vamestyle.posture import _align_rotation   # noqa: E402
from sklearn.cluster import MiniBatchKMeans     # noqa: E402
from sklearn.linear_model import Ridge          # noqa: E402
from sklearn.metrics import (adjusted_mutual_info_score, r2_score,    # noqa: E402
                             normalized_mutual_info_score)

RUN = VAME_STYLE / "runs" / "routeD_posture_vae_50hz"
DS50 = VAME_IMU / "data" / "ds50"
SESS = ["rec_000", "rec_005", "rec_010"]
C = 15


def posture_truth(fs_root, fs, sess, grav_src):
    """逐 session → (pose (T-29,3) [θ,sinφ,cosφ], mu)。"""
    pose, out = [], {}
    for s in sess:
        d = np.load(fs_root / f"{s}.npz")
        g = d["feat9"][:, 3:6].astype(np.float64)      # grav 单位向量
        R = _align_rotation(np.median(g, axis=0))
        g = g @ R.T
        gu = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        theta = np.arccos(np.clip(-gu[:, 2], -1, 1))
        r_xy = np.hypot(gu[:, 0], gu[:, 1])
        sinp = np.where(r_xy > 1e-12, gu[:, 1] / np.maximum(r_xy, 1e-12), 0.0)
        cosp = np.where(r_xy > 1e-12, gu[:, 0] / np.maximum(r_xy, 1e-12), 0.0)
        p = np.stack([theta, sinp, cosp], axis=1)
        pose.append(p[C: len(p) - C + 1])              # 对齐 mu（T-29）
    return pose


def encode_rate(mu, pose):
    """ridge mu→pose held-out R²。"""
    n = len(mu)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: n // 2], idx[n // 2:]
    r = Ridge(alpha=1.0).fit(mu[tr], pose[tr])
    pred = r.predict(mu[te])
    return r2_score(pose[te], pred, multioutput="raw_values")


def recognize_rate(labels, pose, ks=(4, 8)):
    out = {}
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                             batch_size=8192).fit(pose)
        pc = km.labels_
        nmi = adjusted_mutual_info_score(labels, pc)
        # 类内姿态纯度：每状态的主导姿态类占比（加权）
        pur = 0.0
        for st in np.unique(labels):
            m = labels == st
            if m.sum() == 0:
                continue
            pur += m.sum() / len(labels) * np.bincount(pc[m], minlength=k).max() / m.sum()
        out[k] = dict(ami=float(nmi), purity=float(pur))
    return out


def main():
    res = {}
    # ---------- 路线 D（50 Hz）----------
    poseD = posture_truth(DS50, 50.0, SESS, None)
    muD = [np.load(RUN / "outputs" / "mu" / f"mu_{s}.npy") for s in SESS]
    muD_c, poseD_c = np.concatenate(muD), np.concatenate(poseD)
    r2D = encode_rate(muD_c, poseD_c)
    res["routeD"] = dict(
        r2_theta=float(r2D[0]), r2_sinphi=float(r2D[1]), r2_cosphi=float(r2D[2]),
        r2_mean=float(r2D.mean()))
    for K in (27, 36):
        labs = np.concatenate([np.load(RUN / "results" / s /
                                       f"routeD_labels_k{K}.npy") for s in SESS])
        res["routeD"][f"k{K}"] = recognize_rate(labs, poseD_c)
        print(f"[D k{K}] AMI={res['routeD'][f'k{K}'][4]['ami']:.3f} "
              f"纯度={res['routeD'][f'k{K}'][4]['purity']:.3f}", flush=True)

    # ---------- 现行版（25 Hz，对照）----------
    pose25 = posture_truth(VAME_IMU / "data" / "ds25", 25.0, SESS, None)
    mu25 = [np.load(VAME_STYLE / "cache" / f"mu_{s}.npy") for s in SESS]
    mu25_c, pose25_c = np.concatenate(mu25), np.concatenate(pose25)
    r2o = encode_rate(mu25_c, pose25_c)
    res["current"] = dict(
        r2_theta=float(r2o[0]), r2_sinphi=float(r2o[1]), r2_cosphi=float(r2o[2]),
        r2_mean=float(r2o.mean()))
    labs = np.concatenate([np.load(VAME_STYLE / "results" / s /
                                   "labels_k36_v12.npy") for s in SESS])
    res["current"]["k36"] = recognize_rate(labs, pose25_c)
    print(f"[现行 k36] AMI={res['current']['k36'][4]['ami']:.3f} "
          f"纯度={res['current']['k36'][4]['purity']:.3f}", flush=True)
    print(f"[编码 R²] 路线D: θ={r2D[0]:.3f} 均值={r2D.mean():.3f} | "
          f"现行版: θ={r2o[0]:.3f} 均值={r2o.mean():.3f}")

    out = RUN / "results" / "posture_audit.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
