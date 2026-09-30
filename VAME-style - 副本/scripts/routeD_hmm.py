# -*- coding: utf-8 -*-
"""路线 D：BIC 粗扫 + 解码 + 代表图（3 条 pilot，50 Hz）。

K ∈ {2,7,12,17,22,27,32,37,42}（用户指定：从 2 起、步长 5），
GaussianHMM full-cov，口径与 vamestyle.states 一致（seed0 取 10 万连续帧块、
sticky κ=0.98 解码、BIC=-2LL+p·ln n、ICL=BIC+2ΣH）。
幂等：models/routeD_hmm_k{K}.joblib 已存在则跳过；扫完自动解码 BIC 最小 K
与 K=36（对齐现行版口径），出代表图（版式同 state_repr，50 Hz 时间轴）。
用法：venv_python scripts/routeD_hmm.py          # 扫缺失的 K
      venv_python scripts/routeD_hmm.py decode   # 扫完后再解码+绘图
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

from vamestyle.states import sample_block, n_params, icl, sticky_decode   # noqa: E402

RUN = VAME_STYLE / "runs" / "routeD_posture_vae_50hz"
DS50 = VAME_IMU / "data" / "ds50"
SESS = ["rec_000", "rec_005", "rec_010"]
KS = [2, 7, 12, 17, 22, 27, 32, 37, 42]
N_SUB, SEED, KAPPA = 100000, 0, 0.98
N_ITER = 30
SWEEP_CSV = RUN / "results" / "routeD_k_sweep.csv"
BUDGET = 250
N_SEG, MIN_SEG = 20, 8
WIN_PLOT = 3.0


def load_zf():
    zf = {}
    for s in SESS:
        z = np.load(RUN / "outputs" / f"zfeat_D_{s}.npy")
        t = np.load(DS50 / f"{s}.npz")["t"]
        assert len(z) == len(t) - 29
        zf[s] = z
    return zf


def fit_one(K, X, blk, models_dir):
    """EM 分块续跑：每调用至少推进 5 轮并落 partial，反复调用直至冻结。"""
    from hmmlearn.hmm import GaussianHMM
    p = models_dir / f"routeD_hmm_k{K}.joblib"
    pp = models_dir / f"routeD_hmm_k{K}_partial.joblib"
    if p.exists():
        return None
    if pp.exists():
        st = joblib.load(pp)
        m = st["model"]
        m.init_params = ""
        total = st["total_iters"]
    else:
        m = GaussianHMM(n_components=K, covariance_type="full",
                        n_iter=5, random_state=SEED, verbose=False)
        total = 0
    while True:
        m.n_iter = 5
        m.fit(blk)
        total += len(m.monitor_.history)
        conv = bool(m.monitor_.converged)
        print(f"[sweep] K={K} EM 累计 {total} 轮 conv={conv}", flush=True)
        if conv or total >= N_ITER:
            ll = m.score(blk)
            bic = -2 * ll + n_params(K, blk.shape[1]) * np.log(len(blk))
            icl_v = bic + 2 * icl(m, blk)
            joblib.dump(dict(k=K, D=blk.shape[1], loglik=ll, bic=bic,
                             icl=icl_v, converged=conv, n_iter=total,
                             model=m), p)
            pp.unlink(missing_ok=True)
            print(f"[sweep] K={K}: BIC={bic:.0f} ICL={icl_v:.0f} "
                  f"conv={conv}", flush=True)
            return dict(k=K, bic=bic, icl=icl_v)
        joblib.dump({"model": m, "total_iters": total}, pp)


def sweep():
    zf = load_zf()
    X = np.concatenate([zf[n] for n in SESS], axis=0)
    blk = sample_block(X, N_SUB, SEED)
    models_dir = RUN / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = SWEEP_CSV.exists()
    t0 = time.time()
    with open(SWEEP_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not new:
            w.writerow(["k", "bic", "icl"])
        for K in KS:
            r = fit_one(K, X, blk, models_dir)
            if r:
                w.writerow([r["k"], round(r["bic"], 1), round(r["icl"], 1)])
                f.flush()
            if time.time() - t0 > BUDGET:
                print("[sweep] 时间守卫：重跑本命令续扫", flush=True)
                return
    print("[sweep] 全部 K 完成 →", SWEEP_CSV)


def plot_state50(sid, segs, up, outdir, cache):
    """版式同 vamestyle.state_repr.plot_state，数据源 ds50。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    ACC_C = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    GYR_C = ["#d62728", "#9467bd", "#8c564b"]
    nrow, ncol = 4, 5
    fig, axes = plt.subplots(nrow * 2, ncol, figsize=(22, 2.1 * nrow * 2))
    fig.suptitle(f"状态 {sid}（路线D 姿态-VAE+旁路，{up:.1f}%）{len(segs)} 段  "
                 f"上=acc(g)±1.5 下=gyro(dps)±450", fontsize=13)
    C = 15
    for j in range(nrow * ncol):
        ax_a = axes[2 * (j // ncol), j % ncol]
        ax_g = axes[2 * (j // ncol) + 1, j % ncol]
        if j >= len(segs):
            ax_a.axis("off"); ax_g.axis("off")
            continue
        conf, name, a, b, tc = segs[j]
        if name not in cache:
            d = np.load(DS50 / f"{name}.npz")
            cache[name] = (d["t"], d["raw6"])
        t, raw6 = cache[name]
        i0, i1 = a + C, b + C
        c = (t[i0] + t[min(i1, len(t) - 1)]) / 2
        m = (t >= c - WIN_PLOT) & (t <= c + WIN_PLOT)
        tt = t[m] - c
        for ch in range(3):
            ax_a.plot(tt, raw6[m, ch], color=ACC_C[ch], lw=0.7, alpha=1.0)
            ax_g.plot(tt, raw6[m, ch + 3], color=GYR_C[ch], lw=0.7, alpha=1.0)
        for ax in (ax_a, ax_g):
            ax.axvspan(t[i0] - c, t[min(i1, len(t) - 1)] - c,
                       color="orange", alpha=0.25, lw=0)
            ax.set_xlim(-WIN_PLOT, WIN_PLOT)
            ax.tick_params(labelsize=6)
        ax_a.set_ylim(-1.5, 1.5)
        ax_g.set_ylim(-450, 450)
        ax_a.set_title(f"{name} @{tc:.1f}s c={conf:.2f}", fontsize=8)
        if j % ncol == 0:
            ax_a.set_ylabel("acc(g)", fontsize=7)
            ax_g.set_ylabel("gyro(dps)", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = outdir / f"state_{sid:02d}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def decode_and_plot(K):
    zf = load_zf()
    rec = joblib.load(RUN / "models" / f"routeD_hmm_k{K}.joblib")
    model = rec["model"]
    res = RUN / "results"
    cands, usage = {}, np.zeros(K, dtype=np.int64)
    t_all = {s: np.load(DS50 / f"{s}.npz")["t"] for s in SESS}
    for name in SESS:
        z = zf[name]
        lab = sticky_decode(model, z, kappa=KAPPA).astype(np.int16)
        d = res / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / f"routeD_labels_k{K}.npy", lab)
        post = model.predict_proba(z)
        conf_t = post[np.arange(len(lab)), lab]
        dd = np.diff(lab)
        starts = np.concatenate([[0], np.nonzero(dd)[0] + 1])
        ends = np.concatenate([starts[1:], [len(lab)]])
        t = t_all[name]
        for a, b in zip(starts, ends):
            if b - a < MIN_SEG:
                continue
            cands.setdefault(int(lab[a]), []).append(
                (float(conf_t[a:b].mean()), name, int(a), int(b),
                 float(t[a + (b - a) // 2 + 15])))
        usage += np.bincount(lab, minlength=K)
        print(f"[decode] {name} 完成", flush=True)
    pct = 100 * usage / usage.sum()
    with open(res / f"routeD_usage_k{K}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "pct"])
        for i in np.argsort(-pct):
            w.writerow([int(i), round(float(pct[i]), 3)])
    outdir = RUN / "figures" / f"state_repr_k{K}"
    outdir.mkdir(parents=True, exist_ok=True)
    cache = {}
    picked = {sid: sorted(lst, key=lambda x: -x[0])[:N_SEG]
              for sid, lst in cands.items()}
    for sid in sorted(picked):
        p = plot_state50(sid, picked[sid], pct[sid], outdir, cache)
        print(f"[repr] 状态 {sid}: {len(picked[sid])} 段 "
              f"usage={pct[sid]:.1f}% -> {p.name}", flush=True)
    summary = dict(K=K, k_eff=int((pct >= 1.0).sum()),
                   max_pct=round(float(pct.max()), 2),
                   usage={int(i): round(float(v), 3) for i, v in enumerate(pct)})
    json.dump(summary, open(res / f"routeD_hmm_k{K}_summary.json", "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"DECODE_DONE K={K} K_eff={summary['k_eff']} 图目录 {outdir}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "decode":
        rows = list(csv.DictReader(open(SWEEP_CSV, encoding="utf-8")))
        best = min(rows, key=lambda r: float(r["bic"]))
        ks = sorted({int(best["k"]), 36})
        print(f"[decode] BIC 最小 K={best['k']}；解码 K={ks}")
        for K in ks:
            decode_and_plot(K)
    else:
        sweep()


if __name__ == "__main__":
    main()
