"""HSMM 状态数 K 扫描（BIC 版）。

基于当前 HSMM 实现，用 BIC 选择最优 K。
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from hsmm_segment import HSMM, sample_train_frames
from utils import load_config, output_dir

TRAIN_FRAMES = 10000   # 训练采样帧数，25 Hz 下约 400 秒
N_ITER = 10
D_MAX_SECONDS = 5.0


def fit_hsmm(X, n_states, d_max, seed=0):
    """在采样数据上训练 HSMM。"""
    kmeans = KMeans(n_clusters=n_states, random_state=seed, n_init=10)
    kmeans.fit(X)
    hsmm = HSMM(n_states, X.shape[1], d_max=d_max)
    hsmm.emission.mean = kmeans.cluster_centers_
    hsmm.fit(X, n_iter=N_ITER)
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
            # enter_val[t, i] = logsumexp_j (alpha[t-1, j] + log_A[j, i])
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
    """计算 HSMM 自由参数个数。"""
    p_init = n_states - 1
    p_trans = n_states * (n_states - 1)  # 无自环
    p_emission = 2 * n_states * n_dims   # 均值 + 方差（diag）
    p_duration = 2 * n_states            # NB 的 r + p
    return p_init + p_trans + p_emission + p_duration


def main():
    cfg = load_config()
    sc = cfg["hmm_scan"]
    hc = cfg["hmm"]

    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError("没有 latent 文件，请先运行 embed.py")

    X_all = np.concatenate(
        [np.load(p)["downstream"] for p in latent_files], axis=0
    )
    X_all = X_all[np.isfinite(X_all).all(axis=1)]

    fs = cfg["preprocess"].get("target_fs", 100.0)
    d_max = int(D_MAX_SECONDS * fs)
    print(f"HSMM BIC 扫描: fs={fs} Hz, D_max={d_max} frames ({D_MAX_SECONDS}s)")

    # 训练用采样数据
    X_train = sample_train_frames(X_all, TRAIN_FRAMES, seed=0)
    T = len(X_train)
    print(f"训练数据: {X_train.shape}, T={T}")

    k_list = sc["k_list"]
    bics, logliks, n_params = [], [], []

    for k in k_list:
        print(f"--- K={k} ---")
        hsmm = fit_hsmm(X_train, k, d_max, seed=0)
        loglik = hsmm_log_likelihood(hsmm, X_train)
        p = count_params(k, X_train.shape[1])
        bic = -2 * loglik + p * np.log(T)

        logliks.append(loglik)
        n_params.append(p)
        bics.append(bic)

        print(f"  loglik={loglik:.1f}  p={p}  BIC={bic:.1f}")

    # 保存
    scan_dir = out / "hsmm_scan_bic"
    scan_dir.mkdir(exist_ok=True)

    results = np.column_stack([k_list, n_params, logliks, bics])
    np.savetxt(
        scan_dir / "bic_scan.csv",
        results,
        delimiter=",",
        header="K,n_params,loglik,BIC",
        comments="",
        fmt="%.2f",
    )

    best_k = k_list[int(np.argmin(bics))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_list, bics, "o-")
    ax.axvline(best_k, color="r", linestyle="--", label=f"BIC min: K={best_k}")
    ax.set_xlabel("n_states (K)")
    ax.set_ylabel("BIC")
    ax.set_title(f"HSMM BIC scan (fs={fs} Hz, T={T})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(scan_dir / "bic_scan.png", dpi=150)
    plt.show()

    print(f"\nBIC 最优 K={best_k}")
    print(f"报告已保存到 {scan_dir}")


if __name__ == "__main__":
    main()