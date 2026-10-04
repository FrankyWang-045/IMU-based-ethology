"""交叉验证似然扫描：按时间块划分训练/验证，选验证似然饱和的 K。

类别数列表复用 config 的 hmm_scan.k_list。
"""

import numpy as np
import matplotlib.pyplot as plt

from hmm_segment import fit_hmm
from utils import load_config, output_dir

N_FOLDS = 5          # 时间块折数


def make_folds(n_frames, n_folds):
    """把序列按时间连续切成 n_folds 块，返回每块的索引数组列表。"""
    bounds = np.linspace(0, n_frames, n_folds + 1).astype(int)
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(n_folds)]


def main():
    cfg = load_config()
    sc = cfg["hmm_scan"]

    # ---------- 数据 ----------
    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))
    X_all = np.concatenate([np.load(p)["downstream"] for p in latent_files],
                           axis=0)
    X_all = X_all[np.isfinite(X_all).all(axis=1)]

    # 抽样加速（沿用粗扫口径；抽样后排序保持时间结构）
    if sc["sample_frames"] and sc["sample_frames"] < len(X_all):
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(X_all), sc["sample_frames"],
                                 replace=False))
        X_all = X_all[idx]

    folds = make_folds(len(X_all), N_FOLDS)
    print(f"CV 数据: {X_all.shape}, {N_FOLDS} 折，每折约 {len(folds[0])} 帧")

    # ---------- 逐 K 交叉验证 ----------
    train_scores, val_scores = [], []
    for k in sc["k_list"]:
        print(f"--- K={k} ---")
        fold_train, fold_val = [], []
        for i in range(N_FOLDS):
            X_val = X_all[folds[i]]
            X_train = np.concatenate(
                [X_all[folds[j]] for j in range(N_FOLDS) if j != i], axis=0)

            model = fit_hmm(X_train, k, sc["n_seeds"], cfg["hmm"]["n_iter"])
            fold_train.append(model.score(X_train) / len(X_train))
            fold_val.append(model.score(X_val) / len(X_val))

        train_scores.append(np.mean(fold_train))
        val_scores.append(np.mean(fold_val))
        print(f"  K={k}  train={train_scores[-1]:.4f}  val={val_scores[-1]:.4f}")

    # ---------- 保存报告 ----------
    scan_dir = out / "hmm_scan"
    scan_dir.mkdir(exist_ok=True)
    np.savetxt(scan_dir / "cv_scan.csv",
               np.column_stack([sc["k_list"], train_scores, val_scores]),
               delimiter=",", header="K,train_loglik_per_frame,val_loglik_per_frame",
               comments="", fmt="%.4f")

    best_k = sc["k_list"][int(np.argmax(val_scores))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sc["k_list"], train_scores, "o-", label="train")
    ax.plot(sc["k_list"], val_scores, "s-", label="validation")
    ax.axvline(best_k, color="r", linestyle="--",
               label=f"val max: K={best_k}")
    ax.set_xlabel("n_states (K)")
    ax.set_ylabel("log-likelihood per frame")
    ax.set_title(f"HMM {N_FOLDS}-fold CV scan")
    ax.legend()
    fig.tight_layout()
    fig.savefig(scan_dir / "cv_scan.png", dpi=150)
    plt.show()

    print(f"\n验证似然最优: K={best_k}")
    print(f"报告已保存到 {scan_dir}")


if __name__ == "__main__":
    main()