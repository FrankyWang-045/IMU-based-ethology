"""
针对 manifold_test/latent 的 HDP-HSMM 行为分割。
- 读取 *_latent.npz 中的 "latent" 字段
- 可用指定 recording 训练，指定 recording 推断
- 优先读取 config.yaml 的 hdp_hsmm: 段
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

# 可选 numba 加速；未安装时回退到 numpy
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ---------- 配置读取 ----------

def load_config(path: Path | str) -> dict:
    import yaml

    path = Path(path)
    if not path.exists():
        print(f"[警告] 未找到配置文件 {path}，使用默认参数")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}


def get_nested(cfg: dict, *keys, default=None):
    d = cfg
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d if d is not None else default


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
        weights = np.asarray(weights if weights is not None else durations, dtype=float)
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


class GaussianDiagEmission:
    def __init__(self, n_states: int, n_dims: int):
        self.n_states = n_states
        self.n_dims = n_dims
        self.mean = np.zeros((n_states, n_dims))
        self.var = np.ones((n_states, n_dims))

    def log_prob(self, X: np.ndarray) -> np.ndarray:
        diff = X[:, None, :] - self.mean[None, :, :]
        logp = -0.5 * (
            np.log(2 * np.pi * self.var[None, :, :]) + diff ** 2 / self.var[None, :, :]
        )
        return logp.sum(axis=2)

    def fit(self, X: np.ndarray, gamma: np.ndarray):
        weight = gamma.sum(axis=0, keepdims=True).T + 1e-10
        self.mean = (gamma.T @ X) / weight
        diff = X[:, None, :] - self.mean[None, :, :]
        weighted_sq = gamma[:, :, None] * (diff ** 2)
        self.var = weighted_sq.sum(axis=0) / weight + 1e-4


# ---------- HDP-HSMM ----------

@dataclass
class HDP_HSMM_Result:
    states: np.ndarray
    occupied_states: int
    state_weights: np.ndarray
    n_iter: int
    runtime: float


def viterbi_numpy(X, log_B, prefix, log_P, log_A, log_pi, min_d):
    """纯 numpy Viterbi（兼容性回退）。"""
    T = len(X)
    K = log_P.shape[0]
    D = min(log_P.shape[1], T)

    delta = np.full((T, K), -np.inf, dtype=np.float64)
    back_s = np.zeros((T, K), dtype=np.int32)
    back_d = np.zeros((T, K), dtype=np.int32)
    back_j = np.zeros((T, K), dtype=np.int32)

    enter_val = np.full((T, K), -np.inf, dtype=np.float64)
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

    states = np.zeros(T, dtype=np.int32)
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


if HAS_NUMBA:

    @njit(cache=True)
    def viterbi_numba(T, K, D, prefix, log_P, log_A, log_pi, min_d):
        """numba 编译的 Viterbi 核心循环。"""
        delta = np.full((T, K), -1e300)
        back_s = np.zeros((T, K), np.int32)
        back_d = np.zeros((T, K), np.int32)
        back_j = np.zeros((T, K), np.int32)
        enter_val = np.full((T, K), -1e300)
        enter_j = np.full((T, K), -1, np.int32)

        for k in range(K):
            enter_val[0, k] = log_pi[k]

        for t in range(T):
            if t > 0:
                for j in range(K):
                    best = -1e300
                    best_i = -1
                    for i in range(K):
                        if i == j:
                            continue
                        val = log_A[i, j] + delta[t - 1, i]
                        if val > best:
                            best = val
                            best_i = i
                    enter_val[t, j] = best
                    enter_j[t, j] = best_i

            d_max_t = D if D < t + 1 else t + 1
            for k in range(K):
                best_val = -1e300
                best_d = 1
                best_s = t
                best_j = -1
                for d in range(1, d_max_t + 1):
                    if d < min_d:
                        continue
                    s = t - d + 1
                    val = enter_val[s, k] + log_P[k, d - 1] + prefix[s + d, k] - prefix[s, k]
                    if val > best_val:
                        best_val = val
                        best_d = d
                        best_s = s
                        best_j = enter_j[s, k]
                delta[t, k] = best_val
                back_d[t, k] = best_d
                back_s[t, k] = best_s
                back_j[t, k] = best_j

        states = np.zeros(T, np.int32)
        last_i = 0
        best = delta[T - 1, 0]
        for k in range(1, K):
            if delta[T - 1, k] > best:
                best = delta[T - 1, k]
                last_i = k

        t = T - 1
        while t >= 0:
            s = back_s[t, last_i]
            d = back_d[t, last_i]
            for i in range(s, t + 1):
                states[i] = last_i
            prev_j = back_j[t, last_i]
            if prev_j == -1:
                break
            last_i = prev_j
            t = s - 1
        return states


class HDP_HSMM:
    def __init__(
        self,
        K_max: int,
        n_dims: int,
        d_max: int,
        alpha: float = 1.0,
        gamma: float = 1.0,
        duration_kind: str = "nb",
        use_numba: bool = True,
    ):
        self.K_max = K_max
        self.n_dims = n_dims
        self.d_max = d_max
        self.alpha = alpha
        self.gamma = gamma
        self.use_numba = use_numba and HAS_NUMBA

        if self.use_numba:
            print("[信息] 使用 numba 加速 Viterbi")
        else:
            print("[信息] 未安装 numba，使用 numpy Viterbi（较慢，建议 pip install numba）")

        self.state_weights = np.ones(K_max) / K_max
        self.emission = GaussianDiagEmission(K_max, n_dims)
        if duration_kind == "nb":
            self.duration = [NegativeBinomialDuration() for _ in range(K_max)]
        else:
            raise ValueError(f"未知时长分布: {duration_kind}")
        self.log_P = self._compute_log_P()

    def _compute_log_P(self) -> np.ndarray:
        return np.stack([d.log_pmf(self.d_max) for d in self.duration], axis=0)

    def _segment_states(self, states: np.ndarray) -> List[Tuple[int, int, int]]:
        if len(states) == 0:
            return []
        segments = []
        start = 0
        cur = int(states[0])
        for t in range(1, len(states)):
            if states[t] != cur:
                segments.append((cur, start, t - start))
                start = t
                cur = int(states[t])
        segments.append((cur, start, len(states) - start))
        return segments

    def _viterbi(self, X: np.ndarray, min_d: int = 1) -> np.ndarray:
        T = len(X)
        K = self.K_max
        D = min(self.d_max, T)

        log_B = self.emission.log_prob(X)
        prefix = np.concatenate([np.zeros((1, K)), np.cumsum(log_B, axis=0)], axis=0)

        log_P = self.log_P.copy()
        if min_d > 1:
            log_P[:, :min_d] = -np.inf
        log_P = log_P - logsumexp(log_P, axis=1, keepdims=True)

        log_A = np.log(np.ones((K, K)) / (K - 1) + 1e-10)
        np.fill_diagonal(log_A, -np.inf)
        log_pi = np.log(self.state_weights + 1e-10)

        if self.use_numba:
            return viterbi_numba(T, K, D, prefix, log_P, log_A, log_pi, min_d)
        else:
            return viterbi_numpy(X, log_B, prefix, log_P, log_A, log_pi, min_d)

    def _m_step(self, X: np.ndarray, states: np.ndarray):
        T = len(X)
        segments = self._segment_states(states)

        gamma = np.zeros((T, self.K_max))
        gamma[np.arange(T), states] = 1.0
        self.emission.fit(X, gamma)

        duration_hist = [np.zeros(self.d_max, dtype=float) for _ in range(self.K_max)]
        for state, s, d in segments:
            d_idx = min(d - 1, self.d_max - 1)
            duration_hist[state][d_idx] += 1
        for i in range(self.K_max):
            if duration_hist[i].sum() > 0:
                self.duration[i].fit(duration_hist[i], duration_hist[i])

        counts = np.bincount(states, minlength=self.K_max) + self.alpha / self.K_max
        self.state_weights = counts / counts.sum()

        self.log_P = self._compute_log_P()

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 50,
        min_d: int = 5,
        seed: Optional[int] = None,
    ) -> HDP_HSMM_Result:
        rng = np.random.default_rng(seed)
        T = len(X)

        n_init_clusters = min(self.K_max, max(2, self.K_max // 2))
        kmeans = KMeans(n_clusters=n_init_clusters, random_state=seed, n_init=1)
        labels = kmeans.fit_predict(X)
        states = labels % self.K_max

        print(f"开始训练 HDP-HSMM: K_max={self.K_max}, n_iter={n_iter}, 样本数={T}")

        t0 = time.time()
        for it in range(n_iter):
            self._m_step(X, states)

            if it % 20 == 19:
                noise = rng.random(self.state_weights.shape) * 1e-3
                self.state_weights = self.state_weights + noise
                self.state_weights = self.state_weights / self.state_weights.sum()

            states = self._viterbi(X, min_d=min_d)

            if (it + 1) % 10 == 0 or it == 0:
                elapsed = time.time() - t0
                occupied = np.unique(states).shape[0]
                print(f"  迭代 {it + 1:3d}/{n_iter}, 已用 {elapsed:6.1f}s, 当前占用状态数 {occupied}")

        runtime = time.time() - t0
        occupied = int(np.unique(states).shape[0])

        print(f"训练完成: 总耗时 {runtime:.1f}s, 最终占用状态数 {occupied}")

        return HDP_HSMM_Result(
            states=states,
            occupied_states=occupied,
            state_weights=self.state_weights.copy(),
            n_iter=n_iter,
            runtime=runtime,
        )

    def decode(self, X: np.ndarray, min_d: int = 5) -> np.ndarray:
        return self._viterbi(X, min_d=min_d)


# ---------- 数据与流程 ----------

class StandardScaler:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Scaler 未 fit")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist()}, f)

    def load(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.mean = np.asarray(d["mean"])
        self.std = np.asarray(d["std"])


def find_latent_files(latent_dir: Path) -> List[Path]:
    return sorted(latent_dir.glob("*_latent.npz"))


def filter_files(files: List[Path], rec_filter: Optional[List[str]]) -> List[Path]:
    if rec_filter is None or len(rec_filter) == 0:
        return files
    fset = set(rec_filter)
    return [p for p in files if p.stem.replace("_latent", "") in fset]


def load_latent(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    X = data["latent"]
    mask = np.isfinite(X).all(axis=1)
    return {
        "X": X[mask],
        "time": data["time"][mask],
        "interp_mask": data["interp_mask"][mask] if "interp_mask" in data else np.zeros(mask.sum(), dtype=bool),
        "stem": path.stem.replace("_latent", ""),
    }


def parse_recs(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [r.strip() for r in s.split(",") if r.strip()]


def main():
    default_config = Path(__file__).resolve().parent.parent / "config.yaml"

    parser = argparse.ArgumentParser(description="针对 manifold latent 的 HDP-HSMM")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--latent-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--train-recs", type=str, default=None, help="用于训练的 recording（逗号分隔）")
    parser.add_argument("--infer-recs", type=str, default=None, help="用于推断的 recording（逗号分隔）")
    parser.add_argument("--recs", type=str, default=None, help="同时用于训练和推断的 recording（逗号分隔）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hdp_cfg = cfg.get("hdp_hsmm", {})

    # 读取配置：优先 config，其次默认值
    fs = get_nested(cfg, "preprocess", "target_fs", default=25.0)
    K_max = hdp_cfg.get("K_max", 30)
    d_max_sec = hdp_cfg.get("d_max_sec", 5.0)
    d_max = int(d_max_sec * fs)
    min_d = hdp_cfg.get("min_duration_frames", 5)
    alpha = hdp_cfg.get("alpha", 1.0)
    gamma = hdp_cfg.get("gamma", 1.0)
    n_iter = hdp_cfg.get("n_iter", 50)
    seed = hdp_cfg.get("seed", 0)

    print(f"[配置] fs={fs}, K_max={K_max}, d_max={d_max}帧({d_max_sec}s), min_d={min_d}, "
          f"alpha={alpha}, gamma={gamma}, n_iter={n_iter}, seed={seed}")

    latent_dir = Path(args.latent_dir)
    out_dir = Path(args.output_dir) if args.output_dir else latent_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析 recording 参数
    train_recs = parse_recs(args.train_recs)
    infer_recs = parse_recs(args.infer_recs)
    both_recs = parse_recs(args.recs)
    if both_recs:
        train_recs = both_recs
        infer_recs = both_recs

    all_files = find_latent_files(latent_dir)
    train_files = filter_files(all_files, train_recs)
    infer_files = filter_files(all_files, infer_recs)

    if not train_files:
        raise FileNotFoundError(f"未找到训练用 latent 文件: {latent_dir}, train_recs={train_recs}")
    if not infer_files:
        raise FileNotFoundError(f"未找到推断用 latent 文件: {latent_dir}, infer_recs={infer_recs}")

    print(f"[数据] 找到 {len(all_files)} 个文件，训练用 {len(train_files)} 个，推断用 {len(infer_files)} 个")
    for p in train_files:
        print(f"  训练: {p.stem}")
    for p in infer_files:
        print(f"  推断: {p.stem}")

    # 加载训练数据
    train_records = [load_latent(p) for p in train_files]
    X_train = np.concatenate([r["X"] for r in train_records], axis=0)

    # 标准化
    print(f"[预处理] 标准化训练数据: {X_train.shape}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaler.save(out_dir / "hdp_scaler.json")

    # 训练
    model = HDP_HSMM(
        K_max=K_max,
        n_dims=X_train_scaled.shape[1],
        d_max=d_max,
        alpha=alpha,
        gamma=gamma,
    )
    result = model.fit(X_train_scaled, n_iter=n_iter, min_d=min_d, seed=seed)

    print(f"[结果] 实际占用状态数: {result.occupied_states}")

    with open(out_dir / "hdp_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "result": result}, f)

    # 分别解码推断文件
    for rec_idx, p in enumerate(infer_files, 1):
        rec = load_latent(p)
        length = len(rec["X"])
        print(f"[{rec_idx}/{len(infer_files)}] 推断 {rec['stem']} ({length} 帧)...")
        t_dec = time.time()

        X_rec_scaled = scaler.transform(rec["X"])
        states = model.decode(X_rec_scaled, min_d=min_d)

        print(f"  解码完成，耗时 {time.time() - t_dec:.1f}s")

        out_path = out_dir / f"{rec['stem']}_hdp_states.npz"
        np.savez(
            out_path,
            states=states,
            time=rec["time"],
            interp_mask=rec["interp_mask"],
        )

        occ = np.bincount(states, minlength=K_max) / len(states)
        occ_used = occ[occ > 0]
        print(
            f"  已保存 -> {out_path.name}, "
            f"占用 {np.unique(states).shape[0]} 个状态, "
            f"占有率范围 [{occ_used.min():.4f}, {occ_used.max():.4f}]"
        )

    print("HDP-HSMM 全部完成")


if __name__ == "__main__":
    main()