"""
模型选择、交叉验证与诊断模块。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from scipy.special import logsumexp
from scipy.stats import entropy as scipy_entropy

from hsmm_core import (
    FitResult,
    ScanResult,
    CVResult,
    DiagnosticReport,
    logger,
    FixedLengthSampler,
    HBlockCV,
)
from hsmm_models import HSMM, HDP_HSMM, count_hsmm_params


# ---------- 信息准则 ----------

def bic(loglik: float, n_params: int, n_obs: int) -> float:
    return -2 * loglik + n_params * np.log(max(n_obs, 1))


def aic(loglik: float, n_params: int) -> float:
    return -2 * loglik + 2 * n_params


def icl(loglik: float, n_params: int, posterior: np.ndarray) -> float:
    """
    ICL = BIC + 2 * sum_t H(q_t)
    posterior: (T, K) 归属概率矩阵。
    """
    ent = 0.0
    if posterior is not None and posterior.size > 0:
        for t in range(posterior.shape[0]):
            p = posterior[t]
            if p.sum() > 0:
                p = p / p.sum()
                ent += scipy_entropy(p)
    return bic(loglik, n_params, posterior.shape[0]) + 2 * ent


# ---------- 固定长度扫描器 ----------

class FixedLengthScanner:
    def __init__(
        self,
        X: np.ndarray,
        K_list: Sequence[int],
        d_max: int,
        n_iter: int = 20,
        n_inits: int = 5,
        min_d: int = 5,
        duration_kind: str = "nb",
        seed: Optional[int] = 0,
    ):
        self.X = X
        self.K_list = list(K_list)
        self.d_max = d_max
        self.n_iter = n_iter
        self.n_inits = n_inits
        self.min_d = min_d
        self.duration_kind = duration_kind
        self.seed = seed

    def run(self) -> ScanResult:
        logliks = []
        bics = []
        aics = []
        icls = []
        n_obs = len(self.X)
        n_dims = self.X.shape[1]

        for K in self.K_list:
            logger.info(f"扫描 K={K}")
            model = HSMM(K, n_dims, self.d_max, duration_kind=self.duration_kind)
            result = model.fit(
                self.X,
                n_iter=self.n_iter,
                min_d=self.min_d,
                n_inits=self.n_inits,
                seed=self.seed,
            )
            logliks.append(result.loglik)
            np_ = count_hsmm_params(K, n_dims)
            bics.append(bic(result.loglik, np_, n_obs))
            aics.append(aic(result.loglik, np_))
            # 用硬分割 gamma 作为 posterior 近似
            gamma = np.zeros((n_obs, K))
            gamma[np.arange(n_obs), result.viterbi_states] = 1.0
            icls.append(icl(result.loglik, np_, gamma))

        best_bic_k = self.K_list[int(np.argmin(bics))]
        best_aic_k = self.K_list[int(np.argmin(aics))]
        best_icl_k = self.K_list[int(np.argmin(icls))]

        return ScanResult(
            K_list=self.K_list,
            logliks=logliks,
            bics=bics,
            aics=aics,
            icls=icls,
            best_bic_k=best_bic_k,
            best_aic_k=best_aic_k,
            best_icl_k=best_icl_k,
        )


# ---------- 交叉验证 ----------

class HBlockCrossValidator:
    def __init__(
        self,
        X: np.ndarray,
        K_list: Sequence[int],
        d_max: int,
        n_folds: int = 5,
        buffer: int = 25,
        n_iter: int = 10,
        min_d: int = 5,
        duration_kind: str = "nb",
        seed: Optional[int] = 0,
    ):
        self.X = X
        self.K_list = list(K_list)
        self.d_max = d_max
        self.n_folds = n_folds
        self.buffer = buffer
        self.n_iter = n_iter
        self.min_d = min_d
        self.duration_kind = duration_kind
        self.seed = seed

    def run(self) -> List[CVResult]:
        splitter = HBlockCV(n_folds=self.n_folds, buffer=self.buffer, seed=self.seed)
        results = []
        n_dims = self.X.shape[1]

        for K in self.K_list:
            fold_scores = []
            for train_X, val_X in splitter.splits(self.X):
                model = HSMM(K, n_dims, self.d_max, duration_kind=self.duration_kind)
                model.fit(train_X, n_iter=self.n_iter, min_d=self.min_d, seed=self.seed)
                states = model.viterbi(val_X)
                loglik = model.emission.log_prob(val_X)[
                    np.arange(len(val_X)), states
                ].sum()
                fold_scores.append(float(loglik))
            results.append(
                CVResult(
                    K=K,
                    mean_val_loglik=float(np.mean(fold_scores)),
                    std_val_loglik=float(np.std(fold_scores)),
                    folds=fold_scores,
                )
            )
        return results


# ---------- 诊断实验 ----------

class NestedTest:
    """比较 K1 < K2 两个拟合结果的嵌套关系。"""

    def __init__(self, fit1: FitResult, fit2: FitResult):
        self.fit1 = fit1
        self.fit2 = fit2

    def state_mapping(self) -> Dict[int, int]:
        """K2 每个 state 映射到 K1 中出现最多的 state。"""
        s1 = self.fit1.viterbi_states
        s2 = self.fit2.viterbi_states
        mapping = {}
        for k2 in range(self.fit2.K):
            mask = s2 == k2
            if mask.sum() == 0:
                mapping[k2] = -1
            else:
                mapping[k2] = int(np.bincount(s1[mask]).argmax())
        return mapping

    def is_refinement(self) -> bool:
        """K2 的每个 state 都映射到 K1 的单一 state（无新模式）。"""
        mapping = self.state_mapping()
        return all(v >= 0 for v in mapping.values())

    def report(self) -> DiagnosticReport:
        return DiagnosticReport(
            name="nested_test",
            data={
                "K1": self.fit1.K,
                "K2": self.fit2.K,
                "mapping": self.state_mapping(),
                "is_refinement": self.is_refinement(),
            },
        )


class OccupancyTest:
    def __init__(self, fit: FitResult, threshold: float = 0.01):
        self.fit = fit
        self.threshold = threshold

    def report(self) -> DiagnosticReport:
        occ = self.fit.state_occupancy
        rare = [i for i, o in enumerate(occ) if o < self.threshold]
        return DiagnosticReport(
            name="occupancy_test",
            data={
                "K": self.fit.K,
                "occupancy": occ.tolist(),
                "rare_states": rare,
                "threshold": self.threshold,
            },
        )


class DurationTailTest:
    def __init__(self, fit: FitResult, d_max: int):
        self.fit = fit
        self.d_max = d_max

    def report(self) -> DiagnosticReport:
        empirical = self.fit.empirical_durations or {}
        fitted = self.fit.fitted_durations or {}
        summaries = {}
        for s, durs in empirical.items():
            if len(durs) == 0:
                continue
            hist, _ = np.histogram(durs, bins=np.arange(1, self.d_max + 2))
            summaries[s] = {
                "empirical_p95": float(np.percentile(durs, 95)),
                "empirical_max": int(max(durs)),
            }
        return DiagnosticReport(
            name="duration_tail_test",
            data={"K": self.fit.K, "summaries": summaries},
        )


class ParametricBootstrap:
    def __init__(
        self,
        K: int,
        d_max: int,
        n_dims: int,
        length_grid: Sequence[int],
        repeats: int = 5,
        n_iter: int = 10,
        min_d: int = 5,
        seed: Optional[int] = 0,
    ):
        self.K = K
        self.d_max = d_max
        self.n_dims = n_dims
        self.length_grid = list(length_grid)
        self.repeats = repeats
        self.n_iter = n_iter
        self.min_d = min_d
        self.seed = seed

    def _simulate(self, model: HSMM, T: int, rng: np.Any) -> np.ndarray:
        """从 HSMM 参数模拟一条序列（简化版：按当前参数采样）。"""
        X = []
        state = rng.choice(self.K, p=model.pi)
        t = 0
        while t < T:
            d_dist = np.exp(model.log_P[state])
            d = min(rng.choice(self.d_max, p=d_dist) + 1, T - t)
            X.append(rng.normal(model.emission.mean[state], np.sqrt(model.emission.var[state]), size=(d, self.n_dims)))
            trans = model.A[state].copy()
            if trans.sum() == 0:
                trans = np.ones(self.K) / self.K
            else:
                trans = trans / trans.sum()
            state = rng.choice(self.K, p=trans)
            t += d
        return np.concatenate(X, axis=0)[:T]

    def run(self, fit: FitResult) -> DiagnosticReport:
        rng = np.random.default_rng(self.seed)
        n_dims = self.n_dims
        model = HSMM(self.K, n_dims, self.d_max)
        # 用 fit 的参数回填
        model.pi = fit.params.get("pi", model.pi)
        model.A = fit.params.get("A", model.A)
        model.emission.mean = fit.params.get("means", model.emission.mean)
        model.emission.var = fit.params.get("vars", model.emission.var)
        model.log_P = model._compute_log_P()

        curves = {L: [] for L in self.length_grid}
        for b in range(self.repeats):
            X_syn = self._simulate(model, max(self.length_grid), rng)
            for L in self.length_grid:
                X_sub = X_syn[:L]
                scanner = FixedLengthScanner(
                    X_sub,
                    K_list=[self.K],
                    d_max=self.d_max,
                    n_iter=self.n_iter,
                    n_inits=1,
                    min_d=self.min_d,
                    seed=self.seed + b,
                )
                sr = scanner.run()
                curves[L].append(sr.logliks[0])

        return DiagnosticReport(
            name="parametric_bootstrap",
            data={
                "K": self.K,
                "length_grid": self.length_grid,
                "synthetic_loglik_median": {
                    L: float(np.median(v)) for L, v in curves.items()
                },
            },
        )


class StabilitySelection:
    def __init__(
        self,
        X: np.ndarray,
        K_list: Sequence[int],
        d_max: int,
        L: int,
        n_bootstrap: int = 100,
        n_iter: int = 10,
        min_d: int = 5,
        seed: Optional[int] = 0,
    ):
        self.X = X
        self.K_list = list(K_list)
        self.d_max = d_max
        self.L = L
        self.n_bootstrap = n_bootstrap
        self.n_iter = n_iter
        self.min_d = min_d
        self.seed = seed

    def run(self) -> DiagnosticReport:
        rng = np.random.default_rng(self.seed)
        n_obs = len(self.X)
        counts = {"BIC": {}, "AIC": {}, "ICL": {}}

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n_obs, size=self.L)
            X_boot = self.X[idx]
            scanner = FixedLengthScanner(
                X_boot,
                self.K_list,
                self.d_max,
                n_iter=self.n_iter,
                n_inits=1,
                min_d=self.min_d,
                seed=int(rng.integers(0, 1e9)),
            )
            sr = scanner.run()
            for crit, k in zip(
                ["BIC", "AIC", "ICL"],
                [sr.best_bic_k, sr.best_aic_k, sr.best_icl_k],
            ):
                counts[crit][k] = counts[crit].get(k, 0) + 1

        return DiagnosticReport(
            name="stability_selection",
            data={
                "L": self.L,
                "n_bootstrap": self.n_bootstrap,
                "selection_frequencies": counts,
            },
        )


# ---------- 决策汇总 ----------

class DecisionAggregator:
    def __init__(self, scan: ScanResult, cv: Optional[List[CVResult]] = None):
        self.scan = scan
        self.cv = cv

    def run(self) -> Dict[str, Any]:
        candidates = {
            self.scan.best_bic_k,
            self.scan.best_aic_k,
            self.scan.best_icl_k,
        }
        if self.cv:
            # 简单肘部：取 CV loglik 增长变缓的 K
            cv_sorted = sorted(self.cv, key=lambda x: x.K)
            mean_scores = [x.mean_val_loglik for x in cv_sorted]
            elbow_k = cv_sorted[int(np.argmax(mean_scores))].K
            candidates.add(elbow_k)

        table = []
        for k in sorted(candidates):
            row = {
                "K": k,
                "is_bic_best": k == self.scan.best_bic_k,
                "is_aic_best": k == self.scan.best_aic_k,
                "is_icl_best": k == self.scan.best_icl_k,
            }
            if self.cv:
                row["cv_mean_loglik"] = next(
                    (x.mean_val_loglik for x in self.cv if x.K == k), None
                )
            table.append(row)

        return {
            "candidate_Ks": sorted(candidates),
            "decision_table": table,
        }