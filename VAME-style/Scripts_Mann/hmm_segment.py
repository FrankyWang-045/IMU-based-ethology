import pickle

import numpy as np
from hmmlearn.hmm import GaussianHMM


LATENT_PATH = "output/A5_C5_C5-c8_latent.npz"
OUT_PATH = "output/A5_C5_C5-c8_hmm.npz"
MODEL_PATH = "output/hmm_model.pkl"

N_STATES = 6          # 状态数 K，先定一个起点
N_SEEDS = 5 


def fit_hmm(X, n_states, n_seeds=5, n_iter=100):
    """多随机种子训练 HMM，返回对数似然最高的模型。"""
    best_model, best_score = None, -np.inf

    for seed in range(n_seeds):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,            # EM 最大迭代次数
            random_state=seed,
            verbose=False,
        )
        
        model.fit(X)  
        score = model.score(X)
        print(f"  seed={seed}  loglik={score:.1f}")
        if score > best_score:
            best_model, best_score = model, score


    print(f"最佳对数似然: {best_score:.1f}")
    return best_model


def main():
    data = np.load(LATENT_PATH)
    X = data["downstream"]                    # (T, 6)
    time = data["time"]
    print(f"特征形状: {X.shape}")

    bad = ~np.isfinite(X).all(axis=1)
    print(f"非法帧: {bad.sum()}")
    X = X[~bad]

    model = fit_hmm(X, N_STATES, N_SEEDS)

    states = model.predict(X)
    probs = model.predict_proba(X)

    np.savez(OUT_PATH, states=states, probs=probs,
         time=time, interp_mask=data["interp_mask"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    occupancy = np.bincount(states, minlength=N_STATES) / len(states)
    for s, p in enumerate(occupancy):
        print(f"状态 {s}: {p:.1%}")



if __name__ == "__main__":
    main()