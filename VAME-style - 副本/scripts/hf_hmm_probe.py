# -*- coding: utf-8 -*-
"""高频旁路 HMM 探针（第 2 步）。双臂同口径对比：
  arm=base : zfeat 28 列（现行定档口径）
  arm=hf   : zfeat 28 列 + 高频候选 6 列 = 34 列
           [hir_acc_z, hir_gyr_x, hir_gyr_y, hir_gyr_z, hi_short, hi_entropy]
拟合：v12 拼接 → seed 0 取 10 万连续帧块 → GaussianHMM full-cov K=36，
      EM 预算 60 轮（与 vamestyle.states 一致），分块续跑（前台 300 s 线内）。
解码：κ=0.98 sticky Viterbi（vamestyle.states.sticky_decode 原函数）。
判读：BIC/ICL（拟合块）、K_eff(≥1%)、最大类占比、median run length。

幂等：probe_hf/partial_<arm>.joblib 续跑；全部冻结后自动解码出对比。
用法：反复运行直至输出 PROBE_DONE
  venv_python scripts/hf_hmm_probe.py
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

WORKSPACE = Path(r"F:\Kimi\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
VAME_IMU = WORKSPACE / "VAME-IMU"
for p in (str(VAME_STYLE), str(VAME_IMU)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vamestyle.dataset import CFG, ROOT                          # noqa: E402
from vamestyle.features import build as build_zfeat              # noqa: E402
from vamestyle.states import (sample_block, n_params, icl,       # noqa: E402
                              sticky_decode)

FS = float(CFG["data"]["fs"])
HMM = CFG["hmm"]
K = 36
N_SUB = int(HMM["n_sub"])
SEED = int(HMM["seed"])
KAPPA = float(HMM["sticky"])
EM_CHUNK = 5
EM_MAX = 60
BUDGET = 250          # 单次调用秒数守卫
OUT = ROOT / "probe_hf"
OUT.mkdir(exist_ok=True)
V12 = [n for n in build_zfeat() if n not in set(CFG["data"].get("exclude", []))]
HF_NAMES = ["hir_acc_z", "hir_gyr_x", "hir_gyr_y", "hir_gyr_z",
            "hi_short", "hi_entropy"]
HF_IDX = [2, 3, 4, 5, 6, 7]      # 在 8 列候选中的位置（见 hf_feature_check）


def robust_z(x):
    med = np.median(x, axis=0)
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    iqr = q75 - q25
    return np.where(iqr > 1e-8, (x - med) / np.where(iqr > 1e-8, iqr, 1.0), 0.0)


def hf_six(dyn):
    """dyn (T,6) → (T,6) 探针用高频列（未标准化）。"""
    WIN = 75
    freqs = np.fft.rfftfreq(WIN, d=1 / FS)
    hi = freqs >= 5.0
    pad = np.pad(dyn, ((WIN // 2, WIN // 2), (0, 0)), mode="edge")
    w = sliding_window_view(pad, WIN, axis=0)
    psd = np.abs(np.fft.rfft(w, axis=2)) ** 2 / WIN
    hir = psd[:, :, hi].sum(axis=2) / (psd.sum(axis=2) + 1e-12)      # (T,6)
    psd_m = psd.mean(axis=1)
    p_hi = psd_m[:, hi]
    p = p_hi / (p_hi.sum(axis=1, keepdims=True) + 1e-12)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(hi.sum())
    WIN2 = 15
    pad2 = np.pad(dyn, ((WIN2 // 2, WIN2 // 2), (0, 0)), mode="edge")
    w2 = sliding_window_view(pad2, WIN2, axis=0)
    psd2 = (np.abs(np.fft.rfft(w2, axis=2)) ** 2 / WIN2).mean(axis=1)
    f2 = np.fft.rfftfreq(WIN2, d=1 / FS)
    short = np.log10(psd2[:, f2 >= 5.0].mean(axis=1) + 1e-12)
    cand8 = np.concatenate([hir, short[:, None], ent[:, None]], axis=1)
    return cand8[:, HF_IDX]


def load_arm(arm):
    """arm='base' → pooled 28 列；'hf' → pooled 34 列。返回 (X, per_session dict)。"""
    zf = build_zfeat()
    per = {}
    cols = []
    for name in V12:
        z = zf[name]
        if arm == "hf":
            d = np.load(VAME_IMU / "data" / "ds25" / f"{name}.npz")
            h = robust_z(hf_six(d["raw6"].astype(np.float32)))[15:15 + len(z)]
            assert len(h) == len(z)
            z = np.concatenate([z, h.astype(np.float32)], axis=1)
        per[name] = z
        cols.append(z)
    return np.concatenate(cols, axis=0), per


def fit_arm(arm, X, t0):
    """分块 EM 续跑；返回 'frozen' | 'partial'（时间守卫）。"""
    from hmmlearn.hmm import GaussianHMM
    blk = sample_block(X, N_SUB, SEED)
    fp = OUT / f"hmm_k{K}_{arm}.joblib"
    pp = OUT / f"partial_{arm}.joblib"
    if fp.exists():
        return "frozen"
    if pp.exists():
        st = joblib.load(pp)
        model = st["model"]
        model.init_params = ""
        total0 = st["total_iters"]
    else:
        model = GaussianHMM(n_components=K, covariance_type=HMM["covariance_type"],
                            n_iter=EM_CHUNK, random_state=SEED, verbose=False)
        total0 = 0
    while time.time() - t0 < BUDGET:
        model.n_iter = EM_CHUNK
        model.fit(blk)
        total0 += len(model.monitor_.history)
        conv = bool(model.monitor_.converged)
        print(f"[probe:{arm}] EM 累计 {total0} 轮 conv={conv}", flush=True)
        if conv or total0 >= EM_MAX:
            ll = model.score(blk)
            bic = -2 * ll + n_params(K, blk.shape[1]) * np.log(len(blk))
            icl_v = bic + 2 * icl(model, blk)
            joblib.dump(dict(arm=arm, k=K, D=blk.shape[1], loglik=ll, bic=bic,
                             icl=icl_v, converged=conv, n_iter=total0,
                             model=model), fp)
            pp.unlink(missing_ok=True)
            print(f"[probe:{arm}] 冻结 BIC={bic:.0f} ICL={icl_v:.0f}", flush=True)
            return "frozen"
        joblib.dump({"model": model, "total_iters": total0}, pp)
    joblib.dump({"model": model, "total_iters": total0}, pp)
    return "partial"


def decode_arm(arm, per):
    rec = joblib.load(OUT / f"hmm_k{K}_{arm}.joblib")
    model = rec["model"]
    usage = np.zeros(K, dtype=np.int64)
    runs = []
    for name, z in per.items():
        lab = sticky_decode(model, z, kappa=KAPPA)
        usage += np.bincount(lab, minlength=K)
        d = np.diff(lab)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [len(lab)]])
        runs.extend((ends - starts).tolist())
    pct = 100 * usage / usage.sum()
    return dict(arm=arm, D=rec["D"], bic=rec["bic"], icl=rec["icl"],
                converged=rec["converged"], n_iter=rec["n_iter"],
                k_eff=int((pct >= 1.0).sum()), max_pct=round(float(pct.max()), 2),
                median_run=int(np.median(runs)),
                usage={int(i): round(float(v), 3) for i, v in enumerate(pct)})


def main():
    t0 = time.time()
    status = {}
    X_hf, per_hf = None, None
    for arm in ("base", "hf"):
        fp = OUT / f"hmm_k{K}_{arm}.joblib"
        if fp.exists():
            status[arm] = "frozen"
            continue
        if time.time() - t0 > BUDGET:
            status[arm] = "queued"
            continue
        if arm == "hf":
            X_hf, per_hf = load_arm("hf")
            X = X_hf
        else:
            X, _ = load_arm("base")
        status[arm] = fit_arm(arm, X, t0)
    print("拟合状态:", status, flush=True)

    if all((OUT / f"hmm_k{K}_{a}.joblib").exists() for a in ("base", "hf")):
        _, per_base = load_arm("base")
        if per_hf is None:
            _, per_hf = load_arm("hf")
        r_base = decode_arm("base", per_base)
        r_hf = decode_arm("hf", per_hf)
        # 冻结基准（现行定档模型）的参照数字
        ref = {}
        swp = ROOT / "results" / "k_sweep_v12.csv"
        if swp.exists():
            for line in open(swp, encoding="utf-8").readlines()[1:]:
                f = line.strip().split(",")
                if f[0] == str(K):
                    ref = dict(bic=float(f[4]), icl=float(f[5]))
        out = dict(K=K, n_sub=N_SUB, seed=SEED, kappa=KAPPA, em_max=EM_MAX,
                   hf_names=HF_NAMES, reference_frozen_k36_v12=ref,
                   base=r_base, hf=r_hf)
        json.dump(out, open(OUT / "probe_result.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        d_bic = r_hf["bic"] - r_base["bic"]
        d_icl = r_hf["icl"] - r_base["icl"]
        print("\n===== 探针对比（K=36，同一拟合块，同 EM 预算） =====")
        print(f"{'指标':<14}{'base(28d)':>14}{'hf(34d)':>14}{'Δ(hf-base)':>14}")
        for key, f in [("bic", "{:.0f}"), ("icl", "{:.0f}"),
                       ("k_eff", "{}"), ("max_pct", "{}"), ("median_run", "{}")]:
            a, b = r_base[key], r_hf[key]
            print(f"{key:<14}{f.format(a):>14}{f.format(b):>14}{f.format(b - a):>14}")
        if ref:
            print(f"\n参照：冻结定档模型 k36_v12 BIC={ref['bic']:.0f} ICL={ref['icl']:.0f}"
                  f"（EM 预算相同，可校验 base 臂复现）")
        print("PROBE_DONE")
    else:
        print("PROBING（重跑本命令续）")


if __name__ == "__main__":
    main()
