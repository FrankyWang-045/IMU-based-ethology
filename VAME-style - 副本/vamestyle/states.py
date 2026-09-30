# -*- coding: utf-8 -*-
"""v4.0 分割层：Sticky-HMM（GaussianHMM full-cov + sticky 解码）。

口径（对齐 v3.0 重建口径，params：n_iter=30, full, sticky=0.98, n_sub=100k, seed=0）：
  拟合：全 session zfeat 拼接后，seed=0 取一条 n_sub 连续帧块（严禁随机抽帧），
        hmmlearn GaussianHMM(covariance_type='full') EM。
  K 扫描：BIC = -2LL + p·ln(n)；ICL = BIC + 2ΣH(后验)（惩罚成分重叠）。
  sticky 解码：A ← κ·I + (1-κ)·A，行归一化 → Viterbi（保证块结构、抑制闪烁）。
  转移矩阵：拟合原貌 T（社区聚类用，sticky 链近似吸收态会踩坑）。

幂等落盘：
  models/hmm_k<K>.joblib        冻结模型（k, model, n_sub, seed, bic, icl, fit_sec）
  results/k_sweep.csv           K 扫描总表（追加式）
  results/<session>/labels_k<K>.npy    帧级标签（int16，长 = zfeat 行数）
  results/bouts_k<K>.csv        全 session bout 段表（游程提取 + 最短时长过滤）
  results/T_k<K>.npy            拟合原貌转移矩阵
  results/usage_k<K>.csv        状态使用率（K_eff = ≥1% 状态数）

用法：
  python -m vamestyle.states fit 39      # 拟合单个 K（幂等，已拟合则跳过）
  python -m vamestyle.states sweep       # 扫 config 里全部缺失的 K
  python -m vamestyle.states decode 39   # sticky 解码 + bout 表 + 使用率
"""
import csv
import sys
import time
from pathlib import Path

import joblib
import numpy as np

from vamestyle.dataset import CFG, ROOT
from vamestyle.features import build as build_zfeat, build_c as build_zfeat_c

HMM = CFG["hmm"]
STICKY = float(HMM["sticky"])
MIN_BOUT = 4          # 最短 bout：4 帧 @25 Hz = 160 ms（过滤解码闪烁）
EM_CHUNK = 5          # 每次调用最多 EM 轮数（前台 300 s 线内）
EM_MAX_ITERS = 60     # 累计 EM 轮数上限（防不收敛死循环）


def _zf(version=None):
    """按版本取 zfeat 缓存：None=旧 12 维旁路（28 列）；'c'=路线 C（29 列）。"""
    return build_zfeat_c() if version == "c" else build_zfeat()


def pooled_zfeat(subset=None, version=None):
    """session zfeat 拼接（按 config 顺序）。

    subset=None：全部 14 条；subset="v12"：排除 data.exclude（下游验证口径，
    2026-09-28 拍板：rec_011/013 安装角离群）。version="c"：路线 C 特征。
    返回 (X, names, lens)。"""
    zf = _zf(version)
    names = list(zf)
    if subset == "v12":
        ex = set(CFG["data"].get("exclude", []))
        names = [n for n in names if n not in ex]
    lens = [len(zf[n]) for n in names]
    return np.concatenate([zf[n] for n in names], axis=0), names, lens


def _suffix(subset, version=None):
    s = "_v12" if subset == "v12" else ""
    return s + "_c" if version == "c" else s


def sample_block(X, n_sub, seed):
    """seed 取一条 n_sub 连续帧块。"""
    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, X.shape[0] - n_sub))
    return X[start:start + n_sub]


def n_params(K, D):
    """GaussianHMM full-cov 自由参数数（初值 π、A、均值、协方差）。"""
    return (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2


def icl(model, X):
    """ICL = BIC + 2Σ_t H(q_t)（后验熵惩罚重叠成分）。"""
    post = model.predict_proba(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(post)
    logp[~np.isfinite(logp)] = 0.0
    ent = -(post * logp).sum()
    return ent


def fit_k(K, n_sub=None, seed=None, subset=None, version=None):
    """拟合单个 K（幂等）。EM 分块续跑：每调用最多 EM_CHUNK 轮，
    中间态存 models/hmm_k<K>[_v12][_c]_partial.joblib，反复调用直至收敛。
    subset="v12" 时拟合池排除 data.exclude；version="c" 用路线 C 特征。"""
    n_sub = n_sub or int(HMM["n_sub"])
    seed = seed or int(HMM["seed"])
    sfx = _suffix(subset, version)
    mp = ROOT / "models" / f"hmm_k{K}{sfx}.joblib"
    mp.parent.mkdir(exist_ok=True)
    if mp.exists():
        rec = joblib.load(mp)
        print(f"[hmm] k={K}{sfx} 已冻结（BIC={rec['bic']:.0f} ICL={rec['icl']:.0f}），跳过")
        return rec

    from hmmlearn.hmm import GaussianHMM
    X, names, lens = pooled_zfeat(subset, version)
    blk = sample_block(X, n_sub, seed)

    pp = ROOT / "models" / f"hmm_k{K}{sfx}_partial.joblib"
    if pp.exists():
        st = joblib.load(pp)
        model = st["model"]
        model.init_params = ""           # 不从数据重新初始化，续跑 EM
        total0 = st["total_iters"]
        print(f"[hmm] k={K}{sfx} 续跑 EM（已完成 {total0} 轮）")
    else:
        model = GaussianHMM(n_components=K,
                            covariance_type=HMM["covariance_type"],
                            n_iter=EM_CHUNK, random_state=seed, verbose=False)
        total0 = 0

    t0 = time.time()
    model.n_iter = EM_CHUNK
    model.fit(blk)
    sec = time.time() - t0
    total_iters = total0 + len(model.monitor_.history)
    conv = bool(model.monitor_.converged)

    ll = model.score(blk)
    bic = -2 * ll + n_params(K, blk.shape[1]) * np.log(len(blk))
    icl_v = bic + 2 * icl(model, blk)
    print(f"[hmm] k={K}{sfx}: EM +{len(model.monitor_.history)} 轮（累计 {total_iters}）"
          f" conv={conv} LL={ll:.0f} BIC={bic:.0f} ICL={icl_v:.0f} [{sec:.0f}s]")

    if conv or total_iters >= EM_MAX_ITERS:
        rec = dict(k=K, n_sub=n_sub, seed=seed, loglik=ll, bic=bic, icl=icl_v,
                   converged=conv, n_iter=total_iters, fit_sec=None)
        joblib.dump(rec | {"model": model}, mp)
        pp.unlink(missing_ok=True)
        _append_sweep(rec, subset, version)
        print(f"[hmm] k={K}{sfx} 冻结 -> {mp.name}")
    else:
        joblib.dump({"model": model, "total_iters": total_iters}, pp)
        print(f"[hmm] k={K}{sfx} 中间态已存（{pp.name}），再次运行本命令续跑")
    return rec if conv or total_iters >= EM_MAX_ITERS else None


def _append_sweep(rec, subset=None, version=None):
    p = ROOT / "results" / f"k_sweep{_suffix(subset, version)}.csv"
    p.parent.mkdir(exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "n_sub", "seed", "loglik", "bic",
                                          "icl", "converged", "n_iter", "fit_sec"])
        if new:
            w.writeheader()
        w.writerow(rec)


def sticky_decode(model, X, kappa=STICKY):
    """A ← κI + (1-κ)A，行归一化 → Viterbi。返回标签 (T,)。"""
    import copy
    m = copy.deepcopy(model)
    A = m.transmat_ * (1 - kappa)
    A[np.diag_indices_from(A)] += kappa
    m.transmat_ = A / A.sum(axis=1, keepdims=True)
    return m.predict(X)


def decode_k(K, subset=None, version=None):
    """sticky 解码 + bout 表 + 使用率 + 原貌 T。subset="v12" 只解码
    排除 data.exclude 后的 12 条；version="c" 用路线 C 特征与模型。"""
    sfx = _suffix(subset, version)
    rec = joblib.load(ROOT / "models" / f"hmm_k{K}{sfx}.joblib")
    model = rec["model"]
    zf = _zf(version)
    names = list(zf)
    if subset == "v12":
        ex = set(CFG["data"].get("exclude", []))
        names = [n for n in names if n not in ex]
    res = ROOT / "results"
    res.mkdir(exist_ok=True)

    rows, usage = [], np.zeros(K, dtype=np.int64)
    for name in names:
        z = zf[name]
        lab = sticky_decode(model, z).astype(np.int16)
        sdir = res / name
        sdir.mkdir(exist_ok=True)
        np.save(sdir / f"labels_k{K}{sfx}.npy", lab)
        # 未过滤 bout 全量记录（最短时长过滤只用于段表视图）
        d = np.diff(lab)
        starts = np.concatenate([[0], np.nonzero(d)[0] + 1])
        ends = np.concatenate([starts[1:], [len(lab)]])
        dur = ends - starts
        keep = dur >= MIN_BOUT
        t0 = 15 / CFG["data"]["fs"]   # zfeat 行 i ↔ 原始帧 i+15 → 秒
        for s, e in zip(starts[keep], ends[keep]):
            rows.append((name, int(s), int(e), int(e - s),
                         round((s + t0), 3), round((e + t0), 3), int(lab[s])))
        usage += np.bincount(lab, minlength=K)
        print(f"[decode] {name}: {len(rows)} bouts 累计")

    with open(res / f"bouts_k{K}{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "start", "end", "dur", "t_start", "t_end", "state"])
        w.writerows(rows)
    np.save(res / f"T_k{K}{sfx}.npy", model.transmat_)
    pct = 100 * usage / usage.sum()
    with open(res / f"usage_k{K}{sfx}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["state", "pct"])
        for i in np.argsort(-pct):
            w.writerow([int(i), round(float(pct[i]), 3)])
    k_eff = int((pct >= 1.0).sum())
    print(f"[decode] k={K}{sfx}: bout 表 {len(rows)} 段；K_eff(≥1%)={k_eff}；"
          f"最大类 {pct.max():.1f}%")


def refine(subset="v12", budget=240):
    """细扫 k_refine_range 内全部整数 K（幂等 + 时间守卫）：
    反复调用本命令直至全部冻结。"""
    lo, hi = HMM["k_refine_range"]
    t0 = time.time()
    for K in range(lo, hi + 1):
        if (ROOT / "models" / f"hmm_k{K}{_suffix(subset)}.joblib").exists():
            continue
        if time.time() - t0 > budget:
            print(f"[refine] 时间守卫触发，剩余 K 下次继续")
            return
        fit_k(K, subset=subset)
    print("[refine] 全部冻结 REFINE_DONE")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    version = sys.argv[4] if len(sys.argv) > 4 else None
    if cmd == "fit":
        fit_k(int(sys.argv[2]), subset=sys.argv[3] if len(sys.argv) > 3 else None,
              version=version)
    elif cmd == "sweep":
        for K in HMM["k_candidates"]:
            fit_k(K)
    elif cmd == "refine":
        refine(subset=HMM.get("subset", "v12"))
    elif cmd == "decode":
        decode_k(int(sys.argv[2]),
                 subset=sys.argv[3] if len(sys.argv) > 3 else None,
                 version=version)
    else:
        raise SystemExit(f"未知命令 {cmd}")
