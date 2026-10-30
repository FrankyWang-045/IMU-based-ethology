"""
HSMM 基础设施与数据层。
"""
from __future__ import annotations

import json
import logging
import pickle
import hashlib
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Sequence

import numpy as np
import yaml
from scipy.special import logsumexp


# ---------- 结构化对象 ----------

@dataclass
class FitResult:
    K: int
    loglik: float
    n_params: int
    params: Dict[str, Any]
    posterior: Optional[np.ndarray] = None
    viterbi_states: Optional[np.ndarray] = None
    segments: Optional[List[Tuple[int, int, int]]] = None
    state_occupancy: Optional[np.ndarray] = None
    empirical_durations: Optional[Dict[int, List[int]]] = None
    fitted_durations: Optional[Dict[int, Any]] = None
    converged: bool = True
    n_iter: int = 0
    runtime: float = 0.0
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ["posterior", "viterbi_states", "state_occupancy"]:
            v = d.get(k)
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d


@dataclass
class ScanResult:
    K_list: List[int]
    logliks: List[float]
    bics: List[float]
    aics: List[float]
    icls: List[float]
    best_bic_k: int
    best_aic_k: int
    best_icl_k: int

    def to_df_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "K": k,
                "loglik": ll,
                "BIC": bic,
                "AIC": aic,
                "ICL": icl,
            }
            for k, ll, bic, aic, icl in zip(
                self.K_list, self.logliks, self.bics, self.aics, self.icls
            )
        ]


@dataclass
class CVResult:
    K: int
    mean_val_loglik: float
    std_val_loglik: float
    folds: List[float]


@dataclass
class DiagnosticReport:
    name: str
    data: Dict[str, Any]
    figure_paths: List[Path] = field(default_factory=list)


# ---------- 配置读取 ----------

def load_config(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "hmm" not in cfg:
        raise ValueError("config.yaml 中缺少 hmm 段")
    if "hsmm_selection" not in cfg:
        cfg["hsmm_selection"] = {}
    return cfg


def get_hmm_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("hmm", {})


def get_selection_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("hsmm_selection", {})


# ---------- 日志与缓存 ----------

def setup_logger(name: str = "hsmm", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = setup_logger()


def _hash_args(**kwargs) -> str:
    s = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class FitCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key(self, **kwargs) -> str:
        return _hash_args(**kwargs)

    def path(self, key: str) -> Path:
        return self.cache_dir / f"fit_{key}.pkl"

    def get(self, **kwargs) -> Optional[FitResult]:
        p = self.path(self.key(**kwargs))
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, result: FitResult, **kwargs):
        p = self.path(self.key(**kwargs))
        with open(p, "wb") as f:
            pickle.dump(result, f)


# ---------- 公共小工具 ----------

def safe_logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    return logsumexp(a, axis=axis)


def segment_states(states: np.ndarray) -> List[Tuple[int, int, int]]:
    """把状态序列切分为 (state, start, duration) 列表。"""
    if len(states) == 0:
        return []
    segments = []
    start = 0
    cur = states[0]
    for t in range(1, len(states)):
        if states[t] != cur:
            segments.append((int(cur), start, t - start))
            start = t
            cur = states[t]
    segments.append((int(cur), start, len(states) - start))
    return segments


def state_occupancy(states: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(states, minlength=K) / len(states)


# ---------- 数据读取与预处理 ----------

class LatentReader:
    def __init__(self, exp_dir: Path):
        self.exp_dir = Path(exp_dir)

    def find_files(self, rec_filter: Optional[List[str]] = None) -> List[Path]:
        files = sorted(self.exp_dir.glob("*_latent.npz"))
        if not files:
            out_dir = self.exp_dir.parent
            files = sorted(out_dir.glob("*_latent.npz"))
        if rec_filter is not None:
            fset = set(rec_filter)
            files = [
                p for p in files
                if p.stem.replace("_latent", "") in fset
            ]
        return files

    def load(self, path: Path) -> Dict[str, np.ndarray]:
        data = np.load(path)
        X = data["downstream"]
        mask = np.isfinite(X).all(axis=1)
        return {
            "X": X[mask],
            "time": data["time"][mask] if "time" in data else np.arange(len(X))[mask],
            "interp_mask": data["interp_mask"][mask] if "interp_mask" in data else np.zeros(mask.sum(), dtype=bool),
            "stem": path.stem.replace("_latent", ""),
        }


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


class TemporalSplitter:
    """按连续时间块划分，不打散帧。"""

    def __init__(self, val_ratio: float = 0.2, test_ratio: float = 0.0, buffer: int = 0):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.buffer = buffer

    def split(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        n = len(X)
        n_test = int(n * self.test_ratio)
        n_val = int(n * self.val_ratio)
        n_train = n - n_val - n_test

        train = X[:n_train]
        val_start = n_train + self.buffer
        val_end = val_start + n_val - 2 * self.buffer
        val = X[val_start:val_end]

        test_start = n_train + n_val + self.buffer
        test = X[test_start:] if n_test > 0 else np.array([])

        return {"train": train, "val": val, "test": test}


class FixedLengthSampler:
    def __init__(self, length: int, seed: Optional[int] = 0):
        self.length = length
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if n <= self.length:
            return X
        start = self.rng.integers(0, n - self.length + 1)
        return X[start : start + self.length]

    def sample_many(self, Xs: Sequence[np.ndarray], n_samples: int) -> List[np.ndarray]:
        valid = [i for i, X in enumerate(Xs) if len(X) >= self.length]
        if not valid:
            raise ValueError(f"所有 recording 长度均小于 {self.length}")
        out = []
        for _ in range(n_samples):
            idx = self.rng.choice(valid)
            out.append(self.sample(Xs[idx]))
        return out


class HBlockCV:
    """留出长度为 h 的连续块，两侧剔除 buffer。"""

    def __init__(self, n_folds: int = 5, buffer: int = 25, seed: Optional[int] = 0):
        self.n_folds = n_folds
        self.buffer = buffer
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def splits(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_size = n // self.n_folds
        indices = list(range(self.n_folds))
        self.rng.shuffle(indices)
        pairs = []
        for i in indices:
            start = i * fold_size
            end = start + fold_size
            val = X[max(0, start - self.buffer) : min(n, end + self.buffer)]
            train = np.concatenate([X[: max(0, start - self.buffer)], X[min(n, end + self.buffer) :]], axis=0)
            pairs.append((train, val))
        return pairs