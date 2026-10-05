"""HMM 行为分割：全体文件合并训练一个 HMM，逐文件解码保存。"""

import pickle
from turtle import clone

import numpy as np
from hmmlearn.hmm import GaussianHMM

from utils import load_config, output_dir, latent_path, hmm_path

MODEL_NAME = "hmm_model.pkl"


def fit_hmm(X, n_states, n_seeds, n_iter):
    """多随机种子训练 GaussianHMM，退化 seed 自动跳过。"""
    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=n_iter, random_state=seed, verbose=False)
        try:
            with np.errstate(invalid="ignore", divide="ignore"):
                model.fit(X)
            if not (np.isfinite(model.startprob_).all()
                    and np.isfinite(model.transmat_).all()
                    and np.isfinite(model.means_).all()):
                print(f"  seed={seed}  退化，跳过")
                continue
            score = model.score(X)
        except ValueError as e:
            print(f"  seed={seed}  失败，跳过")
            continue
        print(f"  seed={seed}  loglik={score:.1f}")
        if score > best_score:
            best_score, best_model = score, model
    if best_model is None:
        raise RuntimeError(f"K={n_states} 所有 seed 均训练失败")
    print(f"最佳对数似然: {best_score:.1f}")
    return best_model


def main():
    cfg = load_config()
    hc = cfg["hmm"]

    latent_files = sorted(output_dir(cfg).glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError("没有 latent 文件，请先运行 embed.py")

    # 合并全体文件的 latent 训练一个统一 HMM（保证状态坐标系一致）
    X_all = np.concatenate(
        [np.load(p)["downstream"] for p in latent_files], axis=0)
    bad = ~np.isfinite(X_all).all(axis=1)
    if bad.any():
        print(f"警告: 删除 {bad.sum()} 个非法帧")
        X_all = X_all[~bad]
    print(f"训练特征: {X_all.shape}")

    model = fit_hmm(X_all, hc["n_states"], hc["n_seeds"], hc["n_iter"])

    with open(output_dir(cfg) / MODEL_NAME, "wb") as f:
        pickle.dump(model, f)

    # 逐文件解码保存
    for p in latent_files:
        data = np.load(p)
        X = data["downstream"]
        states = model.predict(X)
        probs = model.predict_proba(X)
        np.savez(hmm_path(cfg, p.stem.replace("_latent", "")),
                 states=states, probs=probs,
                 time=data["time"], interp_mask=data["interp_mask"])
        occ = np.bincount(states, minlength=hc["n_states"]) / len(states)
        print(f"  {p.stem}: 状态占比 {np.round(occ, 3)}")


if __name__ == "__main__":
    main()