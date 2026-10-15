"""HSMM 状态数 K 扫描（BIC 版，分段训练 + 并行 K）。
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from hsmm_segment import HSMM
from utils import load_config, output_dir, latest_experiment_dir


def sample_segments(Xs, segment_length, n_segments, seed):
    """从多个 recording 中随机抽取 n_segments 个独立片段。"""
    rng = np.random.default_rng(seed)
    valid_indices = [i for i, X in enumerate(Xs) if len(X) >= segment_length]
    if not valid_indices:
        raise ValueError(f"所有 recording 都短于 segment_length={segment_length}")

    segments = []
    for _ in range(n_segments):
        idx = rng.choice(valid_indices)
        X = Xs[idx]
        start = rng.integers(0, len(X) - segment_length + 1)
        segments.append(X[start : start + segment_length])
    return segments


def fit_hsmm_segments(segments, n_states, d_max, seed=0, n_iter=5, n_init=5):
    """在多个独立片段上训练 HSMM。"""
    init_sample = np.concatenate(segments, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=n_init)
    kmeans.fit(init_sample)

    hsmm = HSMM(n_states, init_sample.shape[1], d_max=d_max)
    hsmm.emission.mean = kmeans.cluster_centers_
    hsmm.fit_segments(segments, n_iter=n_iter, min_d=5)
    return hsmm


def hsmm_log_likelihood(hsmm, X):
    """用前向算法计算 HSMM 对数似然。"""
    T = len(X)
    K = hsmm.n_states
    D = min(hsmm.d_max, T)

    log_B = hsmm.emission.log_prob(X).astype(np.float32)
    prefix = np.concatenate(
        [np.zeros((1, K), dtype=np.float32), np.cumsum(log_B, axis=0)],
        axis=0,
    )

    log_P = hsmm.log_P[:, :D].astype(np.float32)
    log_A = np.log(hsmm.A + 1e-10).astype(np.float32)
    log_pi = np.log(hsmm.pi + 1e-10).astype(np.float32)

    alpha = np.full((T, K), -np.inf, dtype=np.float32)
    enter_val = np.full((T, K), -np.inf, dtype=np.float32)
    enter_val[0, :] = log_pi

    for t in range(T):
        if t > 0:
            enter_val[t, :] = logsumexp(log_A.T + alpha[t - 1, None, :], axis=1)

        d_max_t = min(D, t + 1)
        ds = np.arange(1, d_max_t + 1)
        ss = t - ds + 1

        enter_val_ss = enter_val[ss, :]
        log_P_ds = log_P[:, ds - 1].T
        seg_loglik_ss = prefix[ss + ds, :] - prefix[ss, :]

        vals = enter_val_ss + log_P_ds + seg_loglik_ss
        alpha[t, :] = logsumexp(vals, axis=0)

    return float(logsumexp(alpha[-1, :]))


def count_params(n_states, n_dims):
    p_init = n_states - 1
    p_trans = n_states * (n_states - 1)
    p_emission = 2 * n_states * n_dims
    p_duration = 2 * n_states
    return p_init + p_trans + p_emission + p_duration


def scan_one_k(k, segments, d_max, seed, n_iter, n_init):
    print(f"--- K={k} ---")
    hsmm = fit_hsmm_segments(segments, k, d_max, seed=seed, n_iter=n_iter, n_init=n_init)
    loglik = sum(hsmm_log_likelihood(hsmm, seg) for seg in segments)
    p = count_params(k, segments[0].shape[1])
    return k, loglik, p


def main():
    cfg = load_config()
    sc = cfg["hmm_scan"]        # 扫描专用配置段
    hc = cfg["hmm"]             # HSMM 通用配置段

    out = output_dir(cfg)

    parser = argparse.ArgumentParser(description="HSMM BIC 扫描")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验目录，如 exp_004 或 004；默认最新实验")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="并行核数，默认使用所有核心")
    args = parser.parse_args()

    if args.exp:
        exp_name = args.exp if str(args.exp).startswith("exp_") else f"exp_{int(args.exp):03d}"
        exp_dir = out / exp_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
    else:
        exp_dir = latest_experiment_dir(cfg)

    print(f"使用实验目录: {exp_dir}")

    latent_files = sorted(exp_dir.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError(f"{exp_dir} 中没有 *_latent.npz")

    Xs = []
    for p in latent_files:
        X = np.load(p)["downstream"]
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) > 0:
            Xs.append(X)

    fs = cfg["preprocess"].get("target_fs", 100.0)
    d_max = int(hc.get("hsmm_d_max_sec", 5.0) * fs)

    # ---------- 所有扫描参数都从 config 读取，默认只作为兜底 ----------
    n_iter = sc.get("hsmm_scan_n_iter", 5)
    segment_length = sc.get("hsmm_scan_segment_length", 500)
    n_segments = sc.get("hsmm_scan_n_segments", 10)
    seed = sc.get("hsmm_scan_seed", 0)
    n_init = sc.get("hsmm_scan_n_init", 5)

    segments = sample_segments(Xs, segment_length, n_segments, seed=seed)
    T_total = sum(len(seg) for seg in segments)
    print(f"HSMM BIC 扫描: fs={fs} Hz, D_max={d_max}, "
          f"segments={n_segments} x {segment_length} 帧, 总 T={T_total}")

    k_list = sc["k_list"]

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(scan_one_k)(k, segments, d_max, seed=seed + k, n_iter=n_iter, n_init=n_init)
        for k in k_list
    )

    k_list_out = []
    n_params = []
    logliks = []
    bics = []

    for k, loglik, p in results:
        bic = -2 * loglik + p * np.log(T_total)
        k_list_out.append(k)
        n_params.append(p)
        logliks.append(loglik)
        bics.append(bic)
        print(f"K={k}: loglik={loglik:.1f}  p={p}  BIC={bic:.1f}")

    scan_dir = exp_dir / "hsmm_scan_bic"
    scan_dir.mkdir(exist_ok=True)

    results_arr = np.column_stack([k_list_out, n_params, logliks, bics])
    np.savetxt(
        scan_dir / "bic_scan.csv",
        results_arr,
        delimiter=",",
        header="K,n_params,loglik,BIC",
        comments="",
        fmt="%.2f",
    )

    best_k = k_list_out[int(np.argmin(bics))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_list_out, bics, "o-")
    ax.axvline(best_k, color="r", linestyle="--", label=f"BIC min: K={best_k}")
    ax.set_xlabel("n_states (K)")
    ax.set_ylabel("BIC")
    ax.set_title(f"HSMM BIC scan (fs={fs} Hz, T={T_total})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(scan_dir / "bic_scan.png", dpi=150)
    plt.show()

    print(f"\nBIC 最优 K={best_k}")
    print(f"报告已保存到 {scan_dir}")


if __name__ == "__main__":
    main()