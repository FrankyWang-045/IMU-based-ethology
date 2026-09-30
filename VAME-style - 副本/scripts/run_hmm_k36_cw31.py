# -*- coding: utf-8 -*-
"""v4.1 对照：路线 C 旁路（13 维）+ win=31（与 VAME 输入窗同尺度）重跑 K=36 单层 HMM。

与 scripts/run_hmm_k36.py 完全同口径（v12 拼接 → seed0 取 10 万连续帧块 →
GaussianHMM full-cov EM 分块续跑 → κ=0.98 sticky 解码 → 置信度 top-20 代表图），
唯一变量：zfeat 由 hand12(75 窗, 原始 grav xyz) 换成 hand_c(31 窗, 垂直零点系
θ/sinφ/cosφ/grav_std)。mu、ckpt、HMM 超参一律不动。
输入：cache/mu_<sess>.npy（vame_25hz_14s 的 mu，复用）
      cache/zfeat_c_w31_<sess>.npy（本脚本幂等构建）
输出：models/hmm_k36_v12_c_w31.joblib
      results/<session>/labels_k36_v12_c_w31.npy
      results/usage_k36_v12_c_w31.csv
      figures/state_repr_c_w31/state_XX.png
      results/hmm_k36_c_w31_summary.json
幂等：反复运行直至 HMM_DONE（EM 分块续跑、解码/选段/绘图已存在则跳过）。
用法：venv_python scripts/run_hmm_k36_cw31.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

WORKSPACE = Path(r"F:\Kimi\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
VAME_IMU = WORKSPACE / "VAME-IMU"
for p in (str(VAME_STYLE), str(VAME_IMU)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vamestyle.dataset import CFG, ROOT, sessions                # noqa: E402
from vamestyle.states import (sample_block, n_params, icl,        # noqa: E402
                              sticky_decode)
from vamestyle.features import build_c as build_zfeat_c           # noqa: E402
from vamestyle.state_repr import plot_state, N_SEG, MIN_SEG       # noqa: E402

HMM = CFG["hmm"]
K = 36
N_SUB = int(HMM["n_sub"])
SEED = int(HMM["seed"])
KAPPA = float(HMM["sticky"])
EM_CHUNK = 5
EM_MAX = 60
BUDGET = 250
TAG = "c_w31"
V12 = [s.name for s in sessions()
       if s.name not in set(CFG["data"].get("exclude", []))]
MODEL_P = ROOT / "models" / f"hmm_k36_v12_{TAG}.joblib"
PARTIAL_P = ROOT / "models" / f"hmm_k36_{TAG}_partial.joblib"


def fit(zf):
    from hmmlearn.hmm import GaussianHMM
    X = np.concatenate([zf[n] for n in V12], axis=0)
    blk = sample_block(X, N_SUB, SEED)
    if PARTIAL_P.exists():
        st = joblib.load(PARTIAL_P)
        model = st["model"]
        model.init_params = ""
        total = st["total_iters"]
        print(f"[hmm] 续跑 EM（已完成 {total} 轮）", flush=True)
    else:
        model = GaussianHMM(n_components=K, covariance_type=HMM["covariance_type"],
                            n_iter=EM_CHUNK, random_state=SEED, verbose=False)
        total = 0
    t0 = time.time()
    while True:
        model.n_iter = EM_CHUNK
        model.fit(blk)
        total += len(model.monitor_.history)
        conv = bool(model.monitor_.converged)
        print(f"[hmm] EM 累计 {total} 轮 conv={conv}", flush=True)
        if conv or total >= EM_MAX:
            ll = model.score(blk)
            bic = -2 * ll + n_params(K, blk.shape[1]) * np.log(len(blk))
            icl_v = bic + 2 * icl(model, blk)
            MODEL_P.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(k=K, D=blk.shape[1], n_sub=N_SUB, seed=SEED,
                             loglik=ll, bic=bic, icl=icl_v, converged=conv,
                             n_iter=total, model=model), MODEL_P)
            PARTIAL_P.unlink(missing_ok=True)
            print(f"[hmm] 冻结 BIC={bic:.0f} ICL={icl_v:.0f} -> {MODEL_P}",
                  flush=True)
            return
        joblib.dump({"model": model, "total_iters": total}, PARTIAL_P)
        if time.time() - t0 > BUDGET:
            print("[hmm] 时间守卫：再次运行本命令续跑", flush=True)
            return


def decode_and_select(zf):
    rec = joblib.load(MODEL_P)
    model = rec["model"]
    res = ROOT / "results"
    cands, usage = {}, np.zeros(K, dtype=np.int64)
    for name in V12:
        z = zf[name]
        lab = sticky_decode(model, z, kappa=KAPPA).astype(np.int16)
        d = res / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"labels_k36_v12_{TAG}.npy", lab)
        post = model.predict_proba(z)
        conf_t = post[np.arange(len(lab)), lab]
        C = CFG["vame"]["time_window"] // 2
        s = sessions([name])[0]
        dd = np.diff(lab)
        starts = np.concatenate([[0], np.nonzero(dd)[0] + 1])
        ends = np.concatenate([starts[1:], [len(lab)]])
        for a, b in zip(starts, ends):
            if b - a < MIN_SEG:
                continue
            sid = int(lab[a])
            cands.setdefault(sid, []).append(
                (float(conf_t[a:b].mean()), name, int(a), int(b),
                 float(s.t[a + (b - a) // 2 + C])))
        usage += np.bincount(lab, minlength=K)
        print(f"[decode] {name} 完成", flush=True)
    pct = 100 * usage / usage.sum()
    with open(res / f"usage_k36_v12_{TAG}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "pct"])
        for i in np.argsort(-pct):
            w.writerow([int(i), round(float(pct[i]), 3)])
    picked = {sid: sorted(lst, key=lambda x: -x[0])[:N_SEG]
              for sid, lst in cands.items()}
    return picked, {i: float(v) for i, v in enumerate(pct)}


def main():
    print("[feat] 构建 zfeat_c_w31（win=31，幂等）", flush=True)
    zf_all = build_zfeat_c(win=31)
    zf = {n: zf_all[n] for n in V12}
    if not MODEL_P.exists():
        fit(zf)
        if not MODEL_P.exists():
            print("HMM_PARTIAL（重跑本命令续）")
            return
    picked, usage_pct = decode_and_select(zf)
    outdir = ROOT / "figures" / f"state_repr_{TAG}"
    outdir.mkdir(parents=True, exist_ok=True)
    for sid in sorted(picked):
        p = plot_state(sid, picked[sid], usage_pct.get(sid, 0.0), outdir)
        print(f"[repr] 状态 {sid}: {len(picked[sid])} 段 "
              f"usage={usage_pct.get(sid, 0):.1f}% -> {p.name}", flush=True)
    k_eff = int((np.array(list(usage_pct.values())) >= 1.0).sum())
    summary = dict(tag=TAG, zfeat="route C hand13, win=31",
                   ckpt="vame_25hz_14s best_model（mu 与现行版相同）",
                   K=K, k_eff=k_eff,
                   max_pct=round(max(usage_pct.values()), 2),
                   usage={int(k): round(v, 3) for k, v in usage_pct.items()})
    json.dump(summary, open(ROOT / "results" / f"hmm_k36_{TAG}_summary.json",
                            "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"HMM_DONE  K_eff={k_eff}  最大类={max(usage_pct.values()):.1f}%"
          f"  图目录 {outdir}")


if __name__ == "__main__":
    main()
