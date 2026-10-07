"""Categorical HMM 行为分割：先对 VAE latent 做 k-means 离散化，再训练 Categorical HMM。"""

import pickle

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from sklearn.cluster import KMeans

from utils import load_config, output_dir, latent_path, hmm_cat_path

MODEL_NAME = "hmm_categorical_model.pkl"

def enforce_min_duration(states, min_frames=10):
    """贪婪合并长度 < min_frames 的状态段，直到不再存在短段。"""
    states = states.copy()
    n = len(states)
    if n < min_frames:
        return states

    changed = True
    while changed:
        changed = False
        n = len(states)
        # 找到所有状态切换点
        changes = np.where(np.diff(states) != 0)[0] + 1
        bounds = np.concatenate([[0], changes, [n]])

        for i in range(len(bounds) - 1):
            start, end = bounds[i], bounds[i + 1]
            seg_len = end - start
            if seg_len >= min_frames:
                continue

            # 左右邻居状态及段长
            left_state = states[start - 1] if start > 0 else None
            right_state = states[end] if end < n else None

            if left_state is None and right_state is None:
                continue
            elif left_state is None:
                new_state = right_state
            elif right_state is None:
                new_state = left_state
            else:
                left_len = bounds[i] - bounds[i - 1] if i > 0 else min_frames
                right_len = bounds[i + 2] - bounds[i + 1] if i + 2 < len(bounds) else min_frames
                new_state = left_state if left_len >= right_len else right_state

            states[start:end] = new_state
            changed = True
            break  # 修改后重新扫描

    return states

def make_sticky_transmat(n_states, strength=10.0):
    return np.eye(n_states) * strength + np.ones((n_states, n_states))


def fit_hmm_categorical(X_symbols, n_states, n_seeds, n_iter, sticky_strength=10.0):
    """多随机种子训练 Sticky CategoricalHMM。"""
    transmat_prior = make_sticky_transmat(n_states, sticky_strength)
    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        model = CategoricalHMM(
            n_components=n_states,
            n_iter=n_iter,
            random_state=seed,
            verbose=False,
        )
        model.transmat_prior_ = transmat_prior
        model.fit(X_symbols)
        score = model.score(X_symbols)
        print(f"  seed={seed}  loglik={score:.1f}")
        if score > best_score:
            best_score, best_model = score, model
    print(f"最佳对数似然: {best_score:.1f}")
    return best_model


def main():
    cfg = load_config()
    hc = cfg["hmm"]
    n_symbols = hc.get("n_symbols", 64)

    latent_files = sorted(output_dir(cfg).glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError("没有 latent 文件，请先运行 embed.py")

    # 合并全部 latent 训练 k-means 码本
    X_all = np.concatenate(
        [np.load(p)["downstream"] for p in latent_files], axis=0
    )
    bad = ~np.isfinite(X_all).all(axis=1)
    if bad.any():
        print(f"警告: 删除 {bad.sum()} 个非法帧")
        X_all = X_all[~bad]
    print(f"训练 latent: {X_all.shape}")

    print(f"训练 k-means 码本: {n_symbols} 个符号")
    kmeans = KMeans(n_clusters=n_symbols, random_state=0, n_init=10)
    symbols_all = kmeans.fit_predict(X_all).reshape(-1, 1)

    # 训练 Categorical HMM
    print(f"训练 Categorical HMM: n_states={hc['n_states']}")
    model = fit_hmm_categorical(
        symbols_all,
        hc["n_states"],
        hc["n_seeds"],
        hc["n_iter"],
        sticky_strength=hc.get("sticky_strength", 10.0),
    )

    with open(output_dir(cfg) / MODEL_NAME, "wb") as f:
        pickle.dump({"model": model, "kmeans": kmeans}, f)

    # 逐文件解码保存
    for p in latent_files:
        data = np.load(p)
        X = data["downstream"]
        symbols = kmeans.predict(X).reshape(-1, 1)
        states = model.predict(symbols)
        probs = model.predict_proba(symbols)

        # 强制最小时长 0.1 s
        states = enforce_min_duration(states, min_frames=10)

        np.savez(
            hmm_cat_path(cfg, p.stem.replace("_latent", "")),
            states=states,
            probs=probs,
            symbols=symbols.squeeze(),
            time=data["time"],
            interp_mask=data["interp_mask"],
        )
        occ = np.bincount(states, minlength=hc["n_states"]) / len(states)
        print(f"  {p.stem}: 状态占比 {np.round(occ, 3)}")


if __name__ == "__main__":
    main()