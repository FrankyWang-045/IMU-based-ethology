"""自研 HSMM 行为分割，带最短时长约束与 sticky 时长加权解码。

支持两种训练模式：
1) 单段训练（旧）：把所有 latent 拼起来随机切一段训练；
2) 分段并行训练（新）：每轮随机抽多个独立片段，分别 E-step 后聚合统计量。
"""

import pickle
import sys
from pathlib import Path
import argparse
import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from utils import load_config, output_dir, latest_experiment_dir

MODEL_NAME = "hsmm_model.pkl"


def log_nb_pmf(d, r, p):
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
        d = np.arange(1, self.d_max + 1)
        log_P = self.log_P.copy()

        if beta != 0.0:
            log_P = log_P + beta * d[None, :]

        if min_d > 1:
            log_P[:, :min_d] = -np.inf

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
        """单段 Viterbi EM 训练。"""
        T = len(X)
        print(f"HSMM 单段训练: T={T}, K={self.n_states}, D_max={self.d_max}, min_d={min_d}")

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

    def fit_segments(self, segments, n_iter=20, min_d=10):
        """多段并行 Viterbi EM：段间不连接，每轮聚合统计量。"""
        n_segs = len(segments)
        print(f"HSMM 分段训练: {n_segs} 段/轮, K={self.n_states}, D_max={self.d_max}, min_d={min_d}")

        train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

        for it in range(n_iter):
            # 跨段聚合统计量
            counts_init = np.zeros(self.n_states)
            trans_counts = np.ones((self.n_states, self.n_states))
            state_counts = np.zeros(self.n_states)
            state_sum = np.zeros((self.n_states, self.n_dims))
            state_sumsq = np.zeros((self.n_states, self.n_dims))
            duration_hist = [np.zeros(self.d_max, dtype=float) for _ in range(self.n_states)]

            total_loglik = 0.0
            total_frames = 0

            for X in segments:
                states = self.viterbi(X, log_P=train_log_P)
                T_seg = len(X)

                total_loglik += self.emission.log_prob(X)[np.arange(T_seg), states].sum()
                total_frames += T_seg

                counts_init[states[0]] += 1

                # emission 统计
                for s in range(self.n_states):
                    mask = states == s
                    n_mask = mask.sum()
                    if n_mask > 0:
                        state_counts[s] += n_mask
                        state_sum[s] += X[mask].sum(axis=0)
                        state_sumsq[s] += (X[mask] ** 2).sum(axis=0)

                # 段内转移与时长（不跨段统计转移）
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

            # M-step
            self.pi = (counts_init + 1.0) / (counts_init.sum() + self.n_states)

            self.A = trans_counts / trans_counts.sum(axis=1, keepdims=True)
            np.fill_diagonal(self.A, 0.0)

            weight = state_counts[:, None] + 1e-10
            self.emission.mean = state_sum / weight
            self.emission.var = np.maximum(
                state_sumsq / weight - (state_sum / weight) ** 2,
                0.0,
            ) + 1e-4

            for i in range(self.n_states):
                if duration_hist[i].sum() > 0:
                    self.duration[i].fit(duration_hist[i], duration_hist[i])

            self.log_P = self._compute_log_P()
            train_log_P = self.get_decode_log_P(beta=0.0, min_d=min_d)

            avg_loglik = total_loglik / total_frames
            print(f"  iter {it + 1}/{n_iter}: avg loglik={avg_loglik:.4f}")

        return self


def sample_train_frames(X, n_frames, seed):
    n = len(X)
    if n_frames is None or n <= n_frames:
        return X
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n - n_frames + 1)
    return X[start : start + n_frames]


def sample_segments(Xs, segment_length, n_segments, seed):
    """从多个 recording 中随机抽取 n_segments 个独立片段。"""
    rng = np.random.default_rng(seed)
    valid_indices = [i for i, X in enumerate(Xs) if len(X) >= segment_length]
    if not valid_indices:
        raise ValueError(f"所有 recording 都短于 segment_length={segment_length}")

    segments = []
    for _ in range(n_segments):
        idx = rng.choice(valid_indices)
        X = Xs[idx]
        start = rng.integers(0, len(X) - segment_length + 1)
        segments.append(X[start : start + segment_length])
    return segments


def main(exp_dir_arg=None, rec_filter=None):
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

    # 分段训练参数
    train_segments_per_round = hc.get("hsmm_train_segments_per_round", None)
    segment_length = hc.get("hsmm_segment_length", None)
    n_rounds = hc.get("hsmm_n_rounds", n_iter)

    out = output_dir(cfg)

    # ---------- 解析 exp_dir ----------
    if exp_dir_arg:
        exp_dir = Path(exp_dir_arg)
        if not exp_dir.is_absolute():
            exp_dir = out / exp_dir
    else:
        exp_dir = latest_experiment_dir(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 加载 latent 文件 ----------
    latent_files = sorted(exp_dir.glob("*_latent.npz"))
    if not latent_files:
        latent_files = sorted(out.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError("没有 latent 文件，请先运行 embed.py")

    # ---------- 按 recording 白名单过滤 ----------
    if rec_filter is not None:
        rec_filter_set = set(rec_filter)
        latent_files = [
            p for p in latent_files
            if p.stem.replace("_latent", "") in rec_filter_set
        ]
        if not latent_files:
            raise FileNotFoundError(
                f"白名单 {list(rec_filter_set)} 没有匹配到任何 *_latent.npz 文件"
            )
        print(f"仅处理指定 recording: {sorted(rec_filter_set)}")

    Xs = []
    lengths = []
    for p in latent_files:
        X = np.load(p)["downstream"]
        X = X[np.isfinite(X).all(axis=1)]
        Xs.append(X)
        lengths.append(len(X))
    X_all = np.concatenate(Xs, axis=0)

    # ---------- 初始化 ----------
    if train_segments_per_round and segment_length:
        init_X = sample_segments(Xs, segment_length, train_segments_per_round, seed=seed)
        init_sample = np.concatenate(init_X, axis=0)
    else:
        init_sample = sample_train_frames(X_all, train_frames, seed=seed)

    kmeans = KMeans(n_clusters=hc["n_states"], random_state=seed, n_init=n_init)
    kmeans.fit(init_sample)

    hsmm = HSMM(hc["n_states"], init_sample.shape[1], d_max=d_max)
    hsmm.emission.mean = kmeans.cluster_centers_

    # ---------- 训练 ----------
    if train_segments_per_round and segment_length:
        print(f"分段训练: {train_segments_per_round} 段 x {segment_length} 帧，共 {n_rounds} 轮")
        segments = sample_segments(Xs, segment_length, train_segments_per_round, seed=seed)
        hsmm.fit_segments(segments, n_iter=n_rounds, min_d=min_duration_frames)
    else:
        print(f"单段训练: {train_frames} 帧，共 {n_iter} 轮")
        X_train = sample_train_frames(X_all, train_frames, seed=seed)
        hsmm.fit(X_train, n_iter=n_iter, min_d=min_duration_frames)

    with open(exp_dir / MODEL_NAME, "wb") as f:
        pickle.dump(hsmm, f)

    # ---------- 解码 ----------
    decode_log_P = hsmm.get_decode_log_P(beta=sticky_beta, min_d=min_duration_frames)
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

def parse_rec_list(s):
    """把逗号分隔的字符串转成 list，空则返回 None。"""
    if not s:
        return None
    return [r.strip() for r in s.split(",") if r.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自研 HSMM 行为分割")
    parser.add_argument(
        "--exp", type=str, default=None,
        help="实验目录（如 exp_032），默认使用最新 exp"
    )
    parser.add_argument(
        "--recs", type=str, default=None,
        help="逗号分隔的 recording 白名单，如 A1_C1_A1-6d,A3_C3_A3-6d；默认全量"
    )
    args = parser.parse_args()

    rec_filter = parse_rec_list(args.recs)
    main(exp_dir_arg=args.exp, rec_filter=rec_filter)