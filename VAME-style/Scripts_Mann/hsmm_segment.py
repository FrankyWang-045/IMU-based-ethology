"""自研 HSMM（Hidden Semi-Markov Model）行为分割，带 sticky 时长加权解码。

- 观测：VAE latent z（6 维连续）
- 发射：高斯分布（diag 协方差）
- 时长：Negative Binomial
- 最大时长 D_max 默认 500 帧（5 s @ 100 Hz）
- 解码：P_decode(d) ∝ P_train(d) * exp(beta * d)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from utils import load_config, output_dir

MODEL_NAME = "hsmm_model.pkl"
D_MAX = 500
TRAIN_FRAMES = 10000
STICKY_BETA = 0.0  # 设为 0 禁用 sticky；可尝试 0.001 ~ 0.01


def log_nb_pmf(d, r, p):
    """负二项分布的对数 PMF，d 为帧数（从 1 开始），k = d-1。"""
    k = d - 1
    return (
        gammaln(k + r)
        - gammaln(k + 1)
        - gammaln(r)
        + k * np.log(np.maximum(1 - p, 1e-10))
        + r * np.log(np.maximum(p, 1e-10))
    )


class NegativeBinomialDuration:
    def __init__(self, r=5.0, p=0.5):
        self.r = float(r)
        self.p = float(p)

    def log_pmf(self, d_max):
        d = np.arange(1, d_max + 1)
        lp = log_nb_pmf(d, self.r, self.p)
        return lp - logsumexp(lp)

    def fit(self, durations, weights=None):
        d = np.arange(1, len(durations) + 1)
        if weights is None:
            weights = durations
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            return
        mean = np.sum(d * weights) / weights.sum()
        var = np.sum(weights * (d - mean) ** 2) / weights.sum()
        if var > mean + 1e-6:
            self.p = np.clip(mean / var, 0.01, 0.99)
            self.r = np.clip(mean ** 2 / (var - mean), 0.1, 1000.0)
        else:
            self.r = 1000.0
            self.p = self.r / (mean + self.r)


class GaussianDiagEmission:
    def __init__(self, n_states, n_dims):
        self.n_states = n_states
        self.n_dims = n_dims
        self.mean = np.zeros((n_states, n_dims))
        self.var = np.ones((n_states, n_dims))

    def log_prob(self, X):
        diff = X[:, None, :] - self.mean[None, :, :]
        logp = -0.5 * (
            np.log(2 * np.pi * self.var[None, :, :])
            + diff ** 2 / self.var[None, :, :]
        )
        return logp.sum(axis=2)

    def fit(self, X, gamma):
        weight = gamma.sum(axis=0, keepdims=True).T + 1e-10
        self.mean = (gamma.T @ X) / weight
        diff = X[:, None, :] - self.mean[None, :, :]
        weighted_sq = gamma[:, :, None] * (diff ** 2)
        self.var = weighted_sq.sum(axis=0) / weight + 1e-4


class HSMM:
    def __init__(self, n_states, n_dims, d_max=D_MAX):
        self.n_states = n_states
        self.n_dims = n_dims
        self.d_max = d_max
        self.pi = np.ones(n_states) / n_states
        self.A = np.ones((n_states, n_states)) / (n_states - 1)
        np.fill_diagonal(self.A, 0.0)
        self.emission = GaussianDiagEmission(n_states, n_dims)
        self.duration = [NegativeBinomialDuration() for _ in range(n_states)]
        self.log_P = self._compute_log_P()

    def _compute_log_P(self):
        return np.stack([d.log_pmf(self.d_max) for d in self.duration], axis=0)

    def get_sticky_log_P(self, beta=STICKY_BETA):
        """对时长分布加权：P_decode(d) ∝ P_train(d) * exp(beta * d)。"""
        if beta == 0.0:
            return self.log_P
        d = np.arange(1, self.d_max + 1)
        boosted = self.log_P + beta * d[None, :]
        return boosted - logsumexp(boosted, axis=1, keepdims=True)

    def _segment_loglik(self, X):
        T = len(X)
        log_B = self.emission.log_prob(X)
        prefix = np.concatenate([np.zeros((1, self.n_states)), np.cumsum(log_B, axis=0)], axis=0)
        max_d = min(self.d_max, T)
        seg_loglik = np.full((T, max_d, self.n_states), -np.inf)
        for d in range(1, max_d + 1):
            seg_loglik[: T - d + 1, d - 1, :] = prefix[d:T + 1, :] - prefix[: T - d + 1, :]
        return seg_loglik
    def viterbi(self, X, log_P=None):
        """向量化 HSMM Viterbi，可传入加权后的 log_P。"""
        if log_P is None:
            log_P = self.log_P

        T = len(X)
        K = self.n_states
        D = min(self.d_max, T)

        seg_loglik = self._segment_loglik(X)  # (T, D, K)
        log_P = log_P[:, :D]                  # (K, D)
        log_A = np.log(self.A + 1e-10)        # (K, K)
        log_pi = np.log(self.pi + 1e-10)      # (K,)

        delta = np.full((T, K), -np.inf)
        back_s = np.zeros((T, K), dtype=int)
        back_d = np.zeros((T, K), dtype=int)
        back_j = np.zeros((T, K), dtype=int)

        # enter_val[s, i] = max_j delta[s-1, j] + log_A[j, i]
        # enter_j[s, i] = argmax_j
        enter_val = np.full((T, K), -np.inf)
        enter_j = np.full((T, K), -1, dtype=int)
        enter_val[0, :] = log_pi

        for t in range(T):
            # 计算当前 t 对应的 enter[t, :]
            if t > 0:
                # candidate[i, j] = delta[t-1, j] + log_A[j, i]
                candidate = log_A.T + delta[t - 1, None, :]
                enter_val[t, :] = candidate.max(axis=1)
                enter_j[t, :] = candidate.argmax(axis=1)

            # 有效时长：d = 1..min(D, t+1)
            d_max_t = min(D, t + 1)
            ds = np.arange(1, d_max_t + 1)
            ss = t - ds + 1  # 对应起点

            # vals[d_idx, i] = enter_val[s, i] + log_P[i, d-1] + seg_loglik[s, d-1, i]
            enter_val_ss = enter_val[ss, :]           # (d_max_t, K)
            log_P_ds = log_P[:, ds - 1].T             # (d_max_t, K)
            seg_loglik_ss = seg_loglik[ss, ds - 1, :] # (d_max_t, K)

            vals = enter_val_ss + log_P_ds + seg_loglik_ss  # (d_max_t, K)

            best_d_idx = np.argmax(vals, axis=0)  # (K,)
            delta[t, :] = vals[best_d_idx, np.arange(K)]
            back_d[t, :] = ds[best_d_idx]
            back_s[t, :] = ss[best_d_idx]
            back_j[t, :] = enter_j[ss[best_d_idx], np.arange(K)]

        # 回溯
        states = np.zeros(T, dtype=int)
        last_i = np.argmax(delta[-1, :])
        t = T - 1
        while t >= 0:
            s = back_s[t, last_i]
            d = back_d[t, last_i]
            states[s : t + 1] = last_i
            prev_j = back_j[t, last_i]
            if prev_j == -1:
                break
            last_i = prev_j
            t = s - 1

        return states

    def fit(self, X, n_iter=20):
        T = len(X)
        print(f"HSMM 训练: T={T}, K={self.n_states}, D_max={self.d_max}")

        for it in range(n_iter):
            states = self.viterbi(X)

            segments = []
            t = 0
            while t < T:
                s = t
                state = states[t]
                while t < T and states[t] == state:
                    t += 1
                segments.append((state, s, t - s))

            counts_init = np.bincount([segments[0][0]], minlength=self.n_states)
            self.pi = (counts_init + 1.0) / (counts_init.sum() + self.n_states)

            trans_counts = np.ones((self.n_states, self.n_states))
            for idx in range(len(segments) - 1):
                i, _, _ = segments[idx]
                j, _, _ = segments[idx + 1]
                if i != j:
                    trans_counts[i, j] += 1
            self.A = trans_counts / trans_counts.sum(axis=1, keepdims=True)
            np.fill_diagonal(self.A, 0.0)

            gamma = np.zeros((T, self.n_states))
            for state, s, d in segments:
                gamma[s : s + d, state] = 1.0
            self.emission.fit(X, gamma)

            durations = [[] for _ in range(self.n_states)]
            for state, s, d in segments:
                durations[state].append(d)

            for i in range(self.n_states):
                if len(durations[i]) == 0:
                    continue
                hist = np.bincount(durations[i], minlength=self.d_max + 1)[1:]
                self.duration[i].fit(hist, hist)
            self.log_P = self._compute_log_P()

            avg_loglik = self.emission.log_prob(X)[np.arange(T), states].mean()
            print(f"  iter {it + 1}/{n_iter}: avg loglik={avg_loglik:.4f}, n_segments={len(segments)}")

        return self


def sample_train_frames(X, n_frames, seed):
    n = len(X)
    if n_frames is None or n <= n_frames:
        return X
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n - n_frames + 1)
    return X[start : start + n_frames]


def main():
    cfg = load_config()
    hc = cfg["hmm"]

    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError("没有 latent 文件，请先运行 embed.py")

    Xs = []
    lengths = []
    for p in latent_files:
        X = np.load(p)["downstream"]
        X = X[np.isfinite(X).all(axis=1)]
        Xs.append(X)
        lengths.append(len(X))
    X_all = np.concatenate(Xs, axis=0)

    X_train = sample_train_frames(X_all, TRAIN_FRAMES, seed=0)
    print(f"训练数据: {X_train.shape}")

    kmeans = KMeans(n_clusters=hc["n_states"], random_state=0, n_init=10)
    labels = kmeans.fit_predict(X_train)

    hsmm = HSMM(hc["n_states"], X_train.shape[1], d_max=D_MAX)
    hsmm.emission.mean = kmeans.cluster_centers_

    hsmm.fit(X_train, n_iter=20)

    with open(out / MODEL_NAME, "wb") as f:
        pickle.dump(hsmm, f)

    # 用 sticky 加权后的时长分布解码
    sticky_beta = hc.get("hsmm_sticky_beta", STICKY_BETA)
    print(f"Sticky 解码 beta={sticky_beta}")
    log_P_sticky = hsmm.get_sticky_log_P(sticky_beta)

    for p, length in zip(latent_files, lengths):
        data = np.load(p)
        X = data["downstream"]
        X = X[np.isfinite(X).all(axis=1)]
        states = hsmm.viterbi(X, log_P=log_P_sticky)

        np.savez(
            out / f"{p.stem.replace('_latent', '')}_hsmm.npz",
            states=states,
            time=data["time"],
            interp_mask=data["interp_mask"],
        )

        occ = np.bincount(states, minlength=hc["n_states"]) / len(states)
        print(f"  {p.stem}: 状态占比 {np.round(occ, 3)}")

    print("HSMM 训练与解码完成")


if __name__ == "__main__":
    main()