"""BIC 扫描最佳 HMM 状态数（粗扫：抽样 + 少 seeds）。"""

import numpy as np
import matplotlib.pyplot as plt

from hmm_segment import fit_hmm
from utils import load_config, output_dir


def count_params(n_states, n_features):
    """diag 协方差高斯 HMM 的自由参数个数。"""
    n_start = n_states - 1                 # 初始概率
    n_trans = n_states * (n_states - 1)    # 转移矩阵（每行和为 1）
    n_emit = 2 * n_states * n_features     # 每状态：d 均值 + d 方差
    return n_start + n_trans + n_emit


def main():
    cfg = load_config()
    sc = cfg["hmm_scan"]

    # ---------- 数据：合并全体 latent，随机抽样加速 ----------
    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))
    X_all = np.concatenate([np.load(p)["downstream"] for p in latent_files],
                           axis=0)
    X_all = X_all[np.isfinite(X_all).all(axis=1)]

    if sc["sample_frames"] and sc["sample_frames"] < len(X_all):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_all), sc["sample_frames"], replace=False)
        X = X_all[np.sort(idx)]            # 排序保持时间顺序（HMM 需要序列结构）
    else:
        X = X_all
    print(f"扫描数据: {X.shape}")

    # ---------- 逐 K 扫描 ----------
    n_params_list, logliks, bics = [], [], []
    T, d = X.shape
    for k in sc["k_list"]:
        print(f"--- K={k} ---")
        model = fit_hmm(X, k, sc["n_seeds"], cfg["hmm"]["n_iter"])

        loglik = model.score(X)
        p = count_params(k, d)
        bic = -2 * loglik + p * np.log(T)
        n_params_list.append(p)
        logliks.append(loglik)
        bics.append(bic)
        print(f"  K={k}  p={p}  loglik={loglik:.1f}  BIC={bic:.1f}")

    # ---------- 保存报告 ----------
    scan_dir = out / "hmm_scan"
    scan_dir.mkdir(exist_ok=True)

    results = np.column_stack([sc["k_list"], n_params_list, logliks, bics])
    np.savetxt(scan_dir / "bic_scan.csv", results, delimiter=",",
               header="K,n_params,loglik,BIC", comments="", fmt="%.2f")

    best_k = sc["k_list"][int(np.argmin(bics))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sc["k_list"], bics, "o-")
    ax.axvline(best_k, color="r", linestyle="--", label=f"BIC min: K={best_k}")
    ax.set_xlabel("n_states (K)")
    ax.set_ylabel("BIC")
    ax.set_title(f"HMM BIC scan (sampled {T} frames)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(scan_dir / "bic_scan.png", dpi=150)
    plt.show()

    print(f"\nBIC 最小值: K={best_k}")
    print(f"报告已保存到 {scan_dir}")


if __name__ == "__main__":
    main()