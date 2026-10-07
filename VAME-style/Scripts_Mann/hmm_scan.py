"""HMM 状态数扫描：带重复随机采样的稳健 CV。"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from hmmlearn.hmm import CategoricalHMM, GaussianHMM
from sklearn.cluster import KMeans

from utils import load_config, output_dir

N_FOLDS = 5
TRAIN_FRAMES = 10000
N_REPLICATES = 5    # 每个 fold 重复采样 5 次，降低随机性
METHOD = "gaussian"


def sample_train_frames(X, n_frames, seed):
    n = len(X)
    if n <= n_frames:
        return X
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n - n_frames + 1)
    return X[start : start + n_frames]


def fit_gaussian_hmm(X, n_states, n_seeds):
    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            tol=1e-4,
            init_params="kmeans",
            random_state=seed,
            verbose=False,
        )
        try:
            model.fit(X)
            score = model.score(X)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_model = score, model
    return best_model


def fit_categorical_hmm(X, n_states, n_symbols, n_seeds):
    kmeans = KMeans(n_clusters=n_symbols, random_state=0, n_init=10)
    symbols = kmeans.fit_predict(X).reshape(-1, 1)

    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        model = CategoricalHMM(
            n_components=n_states,
            n_iter=200,
            random_state=seed,
            verbose=False,
        )
        try:
            model.fit(symbols)
            score = model.score(symbols)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_model = score, model
    return best_model, kmeans


def main():
    cfg = load_config()
    sc = cfg["hmm_scan"]
    hc = cfg["hmm"]

    method = sys.argv[1] if len(sys.argv) > 1 else METHOD
    if method not in ("gaussian", "categorical"):
        raise ValueError("method 必须是 'gaussian' 或 'categorical'")

    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))
    X_all = np.concatenate(
        [np.load(p)["downstream"] for p in latent_files], axis=0
    )
    X_all = X_all[np.isfinite(X_all).all(axis=1)]
    print(f"总 latent: {X_all.shape}，每折 {N_REPLICATES} 次重复截 {TRAIN_FRAMES} 帧")

    k_list = sc["k_list"]
    n_seeds = sc["n_seeds"]
    n_symbols = hc.get("n_symbols", 64)

    train_scores, val_scores = [], []
    for k in k_list:
        print(f"--- K={k} ({method}) ---")
        k_train, k_val = [], []

        fold_size = len(X_all) // N_FOLDS
        for fold in range(N_FOLDS):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < N_FOLDS - 1 else len(X_all)

            X_val = X_all[val_start:val_end]
            X_train_full = np.concatenate(
                [X_all[:val_start], X_all[val_end:]], axis=0
            )

            for rep in range(N_REPLICATES):
                seed = fold * N_REPLICATES + rep
                X_train = sample_train_frames(X_train_full, TRAIN_FRAMES, seed)

                if method == "categorical":
                    model, kmeans = fit_categorical_hmm(
                        X_train, k, n_symbols, n_seeds
                    )
                    if model is None:
                        continue
                    val_symbols = kmeans.predict(X_val).reshape(-1, 1)
                    k_train.append(model.score(X_train) / len(X_train))
                    k_val.append(model.score(val_symbols) / len(val_symbols))
                else:
                    model = fit_gaussian_hmm(X_train, k, n_seeds)
                    if model is None:
                        continue
                    k_train.append(model.score(X_train) / len(X_train))
                    k_val.append(model.score(X_val) / len(X_val))

        train_scores.append(k_train)
        val_scores.append(k_val)
        mean_val = np.mean(k_val)
        print(f"  K={k}  val_mean={mean_val:.4f}  (n={len(k_val)})")

    # 汇总统计
    train_mean = [np.mean(s) for s in train_scores]
    train_std = [np.std(s) for s in train_scores]
    val_mean = [np.mean(s) for s in val_scores]
    val_std = [np.std(s) for s in val_scores]

    # 保存
    scan_dir = out / f"hmm_scan_{method}"
    scan_dir.mkdir(exist_ok=True)
    np.savetxt(
        scan_dir / "cv_scan.csv",
        np.column_stack([k_list, train_mean, train_std, val_mean, val_std]),
        delimiter=",",
        header="K,train_mean,train_std,val_mean,val_std",
        comments="",
        fmt="%.4f",
    )

    # 画图
    best_k = k_list[int(np.argmax(val_mean))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(k_list, train_mean, yerr=train_std, fmt="o-", label="train", capsize=3)
    ax.errorbar(k_list, val_mean, yerr=val_std, fmt="s-", label="validation", capsize=3)
    ax.axvline(best_k, color="r", linestyle="--", label=f"val max: K={best_k}")
    ax.set_xlabel("n_states (K)")
    ax.set_ylabel("log-likelihood per frame")
    ax.set_title(f"HMM CV scan ({method}, {N_REPLICATES} reps/fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(scan_dir / "cv_scan.png", dpi=150)
    plt.show()

    print(f"\n验证似然均值最优 K={best_k}")
    print(f"报告已保存到 {scan_dir}")


if __name__ == "__main__":
    main()