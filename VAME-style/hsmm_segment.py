"""HSMM behavior segmentation with configurable duration prior and sticky decoding.

Reads all parameters from config.yaml instead of hard-coded constants.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from utils import load_config, output_dir, latest_experiment_dir

MODEL_NAME = "hsmm_model.pkl"


def log_nb_pmf(d, r, p):
    """Log PMF of Negative Binomial in terms of number of failures k = d-1."""
    k = d - 1
    return (
        gammaln(k + r)
        - gammaln(k + 1)
        - gammaln(r)
        + k * np.log(np.maximum(1 - p, 1e-10))
        + r * np.log(np.maximum(p, 1e-10))
    )


class NegativeBinomialDuration:
    def __init__(self, r=20.0, p=0.2):
        # Default mean = r(1-p)/p ~ 80 frames @ 25 Hz => ~3.2 s
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
    def __init__(self, n_states, n_dims, d_max):
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

    def get_decode_log_P(self, beta, min_d):
        """Build decoding duration distribution with sticky boost + min-duration constraint."""
        d = np.arange(1, self.d_max + 1)
        log_P = self.log_P.copy()

        # sticky boost: favor longer stays in the same state
        if beta != 0.0:
            log_P = log_P + beta * d[None, :]

        # minimum duration constraint
        if min_d > 1:
            log_P[:, :min_d] = -np.inf

        # normalize per state
        return log_P - logsumexp(log_P, axis=1, keepdims=True)

    def viterbi(self, X, log_P=None):
        if log_P is None:
            log_P = self.log_P

        T = len(X)
        K = self.n_states
        D = min(self.d_max, T)

        log_B = self.emission.log_prob(X).astype(np.float32)
        prefix = np.concatenate(
            [np.zeros((1, K), dtype=np.float32), np.cumsum(log_B, axis=0)],
            axis=0,
        )

        log_P = log_P[:, :D].astype(np.float32)
        log_A = np.log(self.A + 1e-10).astype(np.float32)
        log_pi = np.log(self.pi + 1e-10).astype(np.float32)

        delta = np.full((T, K), -np.inf, dtype=np.float32)
        back_s = np.zeros((T, K), dtype=int)
        back_d = np.zeros((T, K), dtype=int)
        back_j = np.zeros((T, K), dtype=int)

        # enter_val[s, i] = max_j delta[s-1, j] + log_A[j, i]
        enter_val = np.full((T, K), -np.inf, dtype=np.float32)
        enter_j = np.full((T, K), -1, dtype=int)
        enter_val[0, :] = log_pi

        for t in range(T):
            if t > 0:
                candidate = log_A.T + delta[t - 1, None, :]
                enter_val[t, :] = candidate.max(axis=1)
                enter_j[t, :] = candidate.argmax(axis=1)

            d_max_t = min(D, t + 1)
            ds = np.arange(1, d_max_t + 1)
            ss = t - ds + 1

            enter_val_ss = enter_val[ss, :]            # (d_max_t, K)
            log_P_ds = log_P[:, ds - 1].T              # (d_max_t, K)
            seg_loglik_ss = prefix[ss + ds, :] - prefix[ss, :]  # (d_max_t, K)

            vals = enter_val_ss + log_P_ds + seg_loglik_ss  # (d_max_t, K)

            best_d_idx = np.argmax(vals, axis=0)
            delta[t, :] = vals[best_d_idx, np.arange(K)]
            back_d[t, :] = ds[best_d_idx]
            back_s[t, :] = ss[best_d_idx]
            back_j[t, :] = enter_j[ss[best_d_idx], np.arange(K)]

        # backtrack
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

    def fit(self, X, n_iter=20, min_d=10):
        """Viterbi EM training. Training also enforces min-duration for stability."""
        T = len(X)
        print(f"HSMM train: T={T}, K={self.n_states}, D_max={self.d_max}, min_d={min_d}")

        train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

        for it in range(n_iter):
            states = self.viterbi(X, log_P=train_log_P)

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
            train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

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
    fs = cfg["preprocess"]["target_fs"]

    # ---------- 从 config 读取参数 ----------
    d_max_sec = hc.get("hsmm_d_max_sec", 5.0)
    d_max = int(d_max_sec * fs)
    min_duration_frames = hc.get("min_duration_frames", int(0.1 * fs))
    train_frames = hc.get("hsmm_train_frames", 10000)
    sticky_beta = hc.get("hsmm_sticky_beta", 0.0)
    n_iter = hc.get("hsmm_n_iter", 20)
    n_init = hc.get("hsmm_n_init", 10)
    seed = hc.get("hsmm_seed", 0)

    # ---------- 确定实验目录 ----------
    out = output_dir(cfg)
    exp_dir = (out / sys.argv[1]) if len(sys.argv) > 1 \
        else latest_experiment_dir(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 优先从 exp_dir 读 latent，找不到再回退到 output_dir
    latent_files = sorted(exp_dir.glob("*_latent.npz"))
    if not latent_files:
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

    X_train = sample_train_frames(X_all, train_frames, seed=seed)
    print(f"训练数据: {X_train.shape}, d_max={d_max}, "
          f"min_d={min_duration_frames}, sticky_beta={sticky_beta}")

    kmeans = KMeans(n_clusters=hc["n_states"], random_state=seed, n_init=n_init)
    labels = kmeans.fit_predict(X_train)

    hsmm = HSMM(hc["n_states"], X_train.shape[1], d_max=d_max)
    hsmm.emission.mean = kmeans.cluster_centers_

    hsmm.fit(X_train, n_iter=n_iter, min_d=min_duration_frames)

    # 保存模型到 exp_dir
    with open(exp_dir / MODEL_NAME, "wb") as f:
        pickle.dump(hsmm, f)

    # 最终解码
    decode_log_P = hsmm.get_decode_log_P(beta=sticky_beta,
                                          min_d=min_duration_frames)
    print(f"最终解码: min_d={min_duration_frames}, sticky_beta={sticky_beta}")

    for p, length in zip(latent_files, lengths):
        data = np.load(p)
        X = data["downstream"]
        X = X[np.isfinite(X).all(axis=1)]
        states = hsmm.viterbi(X, log_P=decode_log_P)

        np.savez(
            exp_dir / f"{p.stem.replace('_latent', '')}_hsmm.npz",
            states=states,
            time=data["time"],
            interp_mask=data["interp_mask"],
        )

        occ = np.bincount(states, minlength=hc["n_states"]) / len(states)
        print(f"  {p.stem}: 状态占比 {np.round(occ, 3)}")

    print("HSMM 训练与解码完成")


if __name__ == "__main__":
    main()
