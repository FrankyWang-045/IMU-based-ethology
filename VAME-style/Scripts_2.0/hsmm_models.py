"""
HSMM 与 HDP-HSMM 模型层。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from hsmm_core import FitResult, logger, segment_states, state_occupancy


# ---------- 分布族 ----------

def log_nb_pmf(d: np.ndarray, r: float, p: float) -> np.ndarray:
    k = d - 1
    return (
        gammaln(k + r)
        - gammaln(k + 1)
        - gammaln(r)
        + k * np.log(np.maximum(1 - p, 1e-10))
        + r * np.log(np.maximum(p, 1e-10))
    )


class NegativeBinomialDuration:
    def __init__(self, r: float = 20.0, p: float = 0.2):
        self.r = float(r)
        self.p = float(p)

    def log_pmf(self, d_max: int) -> np.ndarray:
        d = np.arange(1, d_max + 1)
        lp = log_nb_pmf(d, self.r, self.p)
        return lp - logsumexp(lp)

    def fit(self, durations: np.ndarray, weights: Optional[np.ndarray] = None):
        """durations 为从 1 开始的时长计数直方图或原始时长列表。"""
        if weights is None:
            weights = durations
        weights = np.asarray(weights, dtype=float)
        if weights.sum() == 0:
            return
        d = np.arange(1, len(durations) + 1)
        mean = np.sum(d * weights) / weights.sum()
        var = np.sum(weights * (d - mean) ** 2) / weights.sum()
        if var > mean + 1e-6:
            self.p = float(np.clip(mean / var, 0.01, 0.99))
            self.r = float(np.clip(mean ** 2 / (var - mean), 0.1, 1000.0))
        else:
            self.r = 1000.0
            self.p = self.r / (mean + self.r)


class PoissonDuration:
    def __init__(self, lam: float = 10.0):
        self.lam = float(lam)

    def log_pmf(self, d_max: int) -> np.ndarray:
        d = np.arange(1, d_max + 1, dtype=float)
        lp = d * np.log(self.lam) - self.lam - gammaln(d + 1)
        return lp - logsumexp(lp)

    def fit(self, durations: np.ndarray, weights: Optional[np.ndarray] = None):
        weights = np.asarray(weights if weights is not None else durations, dtype=float)
        if weights.sum() == 0:
            return
        d = np.arange(1, len(durations) + 1)
        self.lam = float(np.sum(d * weights) / weights.sum()) + 1e-6


class GaussianDiagEmission:
    def __init__(self, n_states: int, n_dims: int):
        self.n_states = n_states
        self.n_dims = n_dims
        self.mean = np.zeros((n_states, n_dims))
        self.var = np.ones((n_states, n_dims))

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.mean[None, :, :]
        logp = -0.5 * (np.log(2 * np.pi * self.var[None, :, :]) + diff ** 2 / self.var[None, :, :])
        return logp.sum(axis=2)

    def fit(self, X: np.ndarray, gamma: np.ndarray):
        weight = gamma.sum(axis=0, keepdims=True).T + 1e-10
        self.mean = (gamma.T @ X) / weight
        diff = X[:, None, :] - self.mean[None, :, :]
        weighted_sq = gamma[:, :, None] * (diff ** 2)
        self.var = weighted_sq.sum(axis=0) / weight + 1e-4


# ---------- 参数计数 ----------

def count_hsmm_params(K: int, n_dims: int, duration_n_params: int = 2) -> int:
    """
    HSMM 参数数：
    - 转移：K * (K - 1)（无自转移）
    - 发射均值：K * n_dims
    - 发射方差：K * n_dims
    - 时长参数：K * duration_n_params
    """
    transitions = K * (K - 1)
    emission = 2 * K * n_dims
    duration = K * duration_n_params
    return transitions + emission + duration


# ---------- 固定 K HSMM ----------

class HSMM:
    def __init__(self, n_states: int, n_dims: int, d_max: int, duration_kind: str = "nb"):
        self.n_states = n_states
        self.n_dims = n_dims
        self.d_max = d_max
        self.duration_kind = duration_kind

        self.pi = np.ones(n_states) / n_states
        self.A = np.ones((n_states, n_states)) / (n_states - 1)
        np.fill_diagonal(self.A, 0.0)

        self.emission = GaussianDiagEmission(n_states, n_dims)
        if duration_kind == "nb":
            self.duration = [NegativeBinomialDuration() for _ in range(n_states)]
        elif duration_kind == "poisson":
            self.duration = [PoissonDuration() for _ in range(n_states)]
        else:
            raise ValueError(f"未知时长分布: {duration_kind}")

        self.log_P = self._compute_log_P()

    def _compute_log_P(self) -> np.ndarray:
        return np.stack([d.log_pmf(self.d_max) for d in self.duration], axis=0)

    def get_decode_log_P(self, beta: float = 0.0, min_d: int = 1) -> np.ndarray:
        d = np.arange(1, self.d_max + 1)
        log_P = self.log_P.copy()
        if beta != 0.0:
            log_P = log_P + beta * d[None, :]
        if min_d > 1:
            log_P[:, :min_d] = -np.inf
        return log_P - logsumexp(log_P, axis=1, keepdims=True)

    def viterbi(self, X: np.ndarray, log_P: Optional[np.ndarray] = None) -> np.ndarray:
        if log_P is None:
            log_P = self.log_P

        T = len(X)
        K = self.n_states
        D = min(self.d_max, T)

        log_B = self.emission.log_prob(X).astype(np.float32)
        prefix = np.concatenate([np.zeros((1, K), dtype=np.float32), np.cumsum(log_B, axis=0)], axis=0)

        log_P = log_P[:, :D].astype(np.float32)
        log_A = np.log(self.A + 1e-10).astype(np.float32)
        log_pi = np.log(self.pi + 1e-10).astype(np.float32)

        delta = np.full((T, K), -np.inf, dtype=np.float32)
        back_s = np.zeros((T, K), dtype=np.int32)
        back_d = np.zeros((T, K), dtype=np.int32)
        back_j = np.zeros((T, K), dtype=np.int32)

        enter_val = np.full((T, K), -np.inf, dtype=np.float32)
        enter_j = np.full((T, K), -1, dtype=np.int32)
        enter_val[0, :] = log_pi

        for t in range(T):
            if t > 0:
                candidate = log_A.T + delta[t - 1, None, :]
                enter_val[t, :] = candidate.max(axis=1)
                enter_j[t, :] = candidate.argmax(axis=1)

            d_max_t = min(D, t + 1)
            ds = np.arange(1, d_max_t + 1)
            ss = t - ds + 1

            enter_val_ss = enter_val[ss, :]
            log_P_ds = log_P[:, ds - 1].T
            seg_loglik_ss = prefix[ss + ds, :] - prefix[ss, :]

            vals = enter_val_ss + log_P_ds + seg_loglik_ss

            best_d_idx = np.argmax(vals, axis=0)
            delta[t, :] = vals[best_d_idx, np.arange(K)]
            back_d[t, :] = ds[best_d_idx]
            back_s[t, :] = ss[best_d_idx]
            back_j[t, :] = enter_j[ss[best_d_idx], np.arange(K)]

        states = np.zeros(T, dtype=int)
        last_i = int(np.argmax(delta[-1, :]))
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

    def _m_step(self, X: np.ndarray, states: np.ndarray):
        T = len(X)
        segments = segment_states(states)

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

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 20,
        min_d: int = 5,
        n_inits: int = 1,
        seed: Optional[int] = None,
    ) -> FitResult:
        rng = np.random.default_rng(seed)
        T = len(X)

        best_result: Optional[FitResult] = None
        train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

        for init_id in range(n_inits):
            if init_id > 0:
                # 随机初始化发射均值
                idx = rng.choice(T, size=self.n_states, replace=False)
                self.emission.mean = X[idx].copy()
                self.emission.var = np.ones_like(self.emission.var)

            t0 = time.time()
            for it in range(n_iter):
                states = self.viterbi(X, log_P=train_log_P)
                self._m_step(X, states)
                self.log_P = self._compute_log_P()
                train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

            states = self.viterbi(X, log_P=train_log_P)
            loglik = self.emission.log_prob(X)[np.arange(T), states].sum()
            segments = segment_states(states)

            occ = state_occupancy(states, self.n_states)
            dur = {i: [] for i in range(self.n_states)}
            for s, start, d in segments:
                dur[s].append(d)

            result = FitResult(
                K=self.n_states,
                loglik=float(loglik),
                n_params=count_hsmm_params(self.n_states, self.n_dims),
                params={
                    "pi": self.pi.copy(),
                    "A": self.A.copy(),
                    "means": self.emission.mean.copy(),
                    "vars": self.emission.var.copy(),
                },
                viterbi_states=states,
                segments=segments,
                state_occupancy=occ,
                empirical_durations=dur,
                n_iter=n_iter,
                runtime=time.time() - t0,
                seed=seed,
            )

            if best_result is None or result.loglik > best_result.loglik:
                best_result = result

        return best_result

    def fit_segments(
        self,
        segments_list: List[np.ndarray],
        n_iter: int = 20,
        min_d: int = 5,
        seed: Optional[int] = None,
    ) -> FitResult:
        rng = np.random.default_rng(seed)
        n_segs = len(segments_list)
        train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

        for it in range(n_iter):
            counts_init = np.zeros(self.n_states)
            trans_counts = np.ones((self.n_states, self.n_states))
            state_counts = np.zeros(self.n_states)
            state_sum = np.zeros((self.n_states, self.n_dims))
            state_sumsq = np.zeros((self.n_states, self.n_dims))
            duration_hist = [np.zeros(self.d_max, dtype=float) for _ in range(self.n_states)]

            total_loglik = 0.0
            total_frames = 0

            for X in segments_list:
                states = self.viterbi(X, log_P=train_log_P)
                T_seg = len(X)
                total_loglik += self.emission.log_prob(X)[np.arange(T_seg), states].sum()
                total_frames += T_seg

                counts_init[states[0]] += 1

                for s in range(self.n_states):
                    mask = states == s
                    n_mask = mask.sum()
                    if n_mask > 0:
                        state_counts[s] += n_mask
                        state_sum[s] += X[mask].sum(axis=0)
                        state_sumsq[s] += (X[mask] ** 2).sum(axis=0)

                t = 0
                while t < T_seg:
                    s = t
                    state = states[t]
                    while t < T_seg and states[t] == state:
                        t += 1
                    d = t - s
                    d_idx = min(d - 1, self.d_max - 1)
                    duration_hist[state][d_idx] += 1
                    if t < T_seg:
                        next_state = states[t]
                        if state != next_state:
                            trans_counts[state, next_state] += 1

            self.pi = (counts_init + 1.0) / (counts_init.sum() + self.n_states)
            self.A = trans_counts / trans_counts.sum(axis=1, keepdims=True)
            np.fill_diagonal(self.A, 0.0)

            weight = state_counts[:, None] + 1e-10
            self.emission.mean = state_sum / weight
            self.emission.var = np.maximum(state_sumsq / weight - (state_sum / weight) ** 2, 0.0) + 1e-4

            for i in range(self.n_states):
                if duration_hist[i].sum() > 0:
                    self.duration[i].fit(duration_hist[i], duration_hist[i])

            self.log_P = self._compute_log_P()
            train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

        # 最终在全段上算一次 loglik（取拼接序列）
        X_all = np.concatenate(segments_list, axis=0)
        states = self.viterbi(X_all, log_P=train_log_P)
        loglik = self.emission.log_prob(X_all)[np.arange(len(X_all)), states].sum()
        return FitResult(
            K=self.n_states,
            loglik=float(loglik),
            n_params=count_hsmm_params(self.n_states, self.n_dims),
            params={"pi": self.pi.copy(), "A": self.A.copy()},
            viterbi_states=states,
            seed=seed,
        )


# ---------- HDP-HSMM（截断 stick-breaking） ----------

class HDP_HSMM:
    """
    轻量截断 HDP-HSMM。
    状态数上限为 K_max，通过 Dirichlet 过程 stick-breaking 先验让实际占用状态数自适应。
    推断：简单的 blocked Gibbs 采样（显式时长）。
    """

    def __init__(
        self,
        K_max: int,
        n_dims: int,
        d_max: int,
        alpha: float = 1.0,
        gamma: float = 1.0,
        duration_kind: str = "nb",
    ):
        self.K_max = K_max
        self.n_dims = n_dims
        self.d_max = d_max
        self.alpha = alpha
        self.gamma = gamma
        self.duration_kind = duration_kind

        # stick-breaking 权重先验
        self.beta_sticks = np.ones(K_max) / K_max
        self.state_weights = np.ones(K_max) / K_max

        self.emission = GaussianDiagEmission(K_max, n_dims)
        if duration_kind == "nb":
            self.duration = [NegativeBinomialDuration() for _ in range(K_max)]
        elif duration_kind == "poisson":
            self.duration = [PoissonDuration() for _ in range(K_max)]

        self.log_P = self._compute_log_P()

    def _compute_log_P(self) -> np.ndarray:
        return np.stack([d.log_pmf(self.d_max) for d in self.duration], axis=0)

    def _sample_states(self, X: np.ndarray, min_d: int = 1) -> np.ndarray:
        """用当前参数做 Viterbi 点估计（Gibbs 第一步近似）。"""
        log_P = self.log_P.copy()
        if min_d > 1:
            log_P[:, :min_d] = -np.inf
            log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)

        # 构造一个固定 K HSMM 用于 Viterbi
        proxy = HSMM(self.K_max, self.n_dims, self.d_max, self.duration_kind)
        proxy.pi = self.state_weights.copy()
        proxy.A = np.ones((self.K_max, self.K_max)) / (self.K_max - 1)
        np.fill_diagonal(proxy.A, 0.0)
        proxy.emission = self.emission
        proxy.duration = self.duration
        proxy.log_P = self.log_P
        return proxy.viterbi(X, log_P=log_P)

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 100,
        min_d: int = 5,
        seed: Optional[int] = None,
    ) -> FitResult:
        rng = np.random.default_rng(seed)
        T = len(X)

        # K-means 初始化
        kmeans = KMeans(n_clusters=min(self.K_max, 10), random_state=seed, n_init=1)
        labels = kmeans.fit_predict(X)
        states = labels % self.K_max

        t0 = time.time()
        for it in range(n_iter):
            # M-step：根据当前状态更新参数
            proxy = HSMM(self.K_max, self.n_dims, self.d_max, self.duration_kind)
            proxy.emission = self.emission
            proxy.duration = self.duration
            proxy._m_step(X, states)

            # 更新 stick-breaking 权重（简化：按出现次数 + DP 浓度）
            counts = np.bincount(states, minlength=self.K_max) + self.alpha / self.K_max
            self.state_weights = rng.dirichlet(counts + self.gamma / self.K_max)

            # E-step：重新解码
            states = self._sample_states(X, min_d=min_d)

        loglik = self.emission.log_prob(X)[np.arange(T), states].sum()
        segments = segment_states(states)
        occ = state_occupancy(states, self.K_max)

        return FitResult(
            K=self.K_max,
            loglik=float(loglik),
            n_params=count_hsmm_params(self.K_max, self.n_dims),
            params={
                "state_weights": self.state_weights.copy(),
                "occupied_states": int(np.unique(states).shape[0]),
            },
            viterbi_states=states,
            segments=segments,
            state_occupancy=occ,
            n_iter=n_iter,
            runtime=time.time() - t0,
            seed=seed,
        )