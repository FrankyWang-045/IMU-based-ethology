"""HMM 行为分割：全体文件合并训练一个 HMM，逐文件解码保存。"""

import pickle

import numpy as np
from hmmlearn.hmm import GaussianHMM

from utils import load_config, output_dir, latent_path, hmm_path

MODEL_NAME = "hmm_model.pkl"


import numpy as np  # 若文件顶部未导入则加上


def make_sticky_transmat(n_states, strength=10.0):
    """构造转移矩阵的 Dirichlet 先验：对角线越大，越鼓励自转移。"""
    return np.eye(n_states) * strength + np.ones((n_states, n_states))

def fit_hmm(X, n_states, n_seeds, n_iter):
    """自由 EM 拟合 full-cov GaussianHMM，训练阶段不加 sticky 约束。"""
    best_model, best_score = None, -np.inf
    for seed in range(n_seeds):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=seed,
            verbose=False,
        )
        model.fit(X)
        score = model.score(X)
        print(f"  seed={seed}  loglik={score:.1f}")
        if score > best_score:
            best_score, best_model = score, model
    print(f"最佳对数似然: {best_score:.1f}")
    return best_model

def make_sticky_decode_transmat(transmat, self_loop=0.98):
    """
    把训练好的转移矩阵改为高自环版本。
    自环概率固定为 self_loop，离环概率按原矩阵的相对比例重新分配。
    """
    n = transmat.shape[0]
    sticky = np.zeros_like(transmat)
    for i in range(n):
        off_diag = transmat[i].copy()
        off_diag[i] = 0.0
        off_sum = off_diag.sum()
        if off_sum > 0:
            sticky[i] = off_diag * (1.0 - self_loop) / off_sum
        else:
            sticky[i, :] = (1.0 - self_loop) / (n - 1)
            sticky[i, i] = 0.0
        sticky[i, i] = self_loop
    return sticky


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

    # 保存原始模型：BIC / CV / 转移统计都用它
    with open(output_dir(cfg) / MODEL_NAME, "wb") as f:
        pickle.dump(model, f)

    # 构造解码专用 sticky 转移矩阵
    self_loop = hc.get("self_loop", 0.98)
    sticky_transmat = make_sticky_decode_transmat(model.transmat_, self_loop)
    print(f"解码用 sticky 自环概率: {self_loop:.2f}")

    # 逐文件用 sticky 转移矩阵做 Viterbi 解码
    for p in latent_files:
        data = np.load(p)
        X = data["downstream"]

        orig_transmat = model.transmat_.copy()
        model.transmat_ = sticky_transmat

        states = model.predict(X)
        probs = model.predict_proba(X)

        model.transmat_ = orig_transmat  # 恢复原始模型

        np.savez(
            hmm_path(cfg, p.stem.replace("_latent", "")),
            states=states,
            probs=probs,
            time=data["time"],
            interp_mask=data["interp_mask"],
        )
        occ = np.bincount(states, minlength=hc["n_states"]) / len(states)
        print(f"  {p.stem}: 状态占比 {np.round(occ, 3)}")


if __name__ == "__main__":
    main()