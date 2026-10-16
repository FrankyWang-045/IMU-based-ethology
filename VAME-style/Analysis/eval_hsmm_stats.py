"""HSMM 统计评估：分割质量、state 一致性、跨文件一致性、时序结构"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap


def load_config():
    import yaml
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_exp(output_root: Path):
    exps = [d for d in output_root.iterdir() if d.is_dir() and re.match(r"exp_\d+", d.name)]
    if not exps:
        raise FileNotFoundError(f"在 {output_root} 下未找到 exp_* 文件夹")
    return sorted(exps, key=lambda d: int(d.name.split("_")[1]))[-1]


def load_pair(exp_dir: Path, stem: str):
    hsmm = np.load(exp_dir / f"{stem}_hsmm.npz")
    latent = np.load(exp_dir / f"{stem}_latent.npz")
    mask = hsmm["interp_mask"].astype(bool)
    return {
        "stem": stem,
        "states": hsmm["states"][mask],
        "time": hsmm["time"][mask],
        "latent": latent["downstream"][mask],
    }


def segment_durations(states, fs):
    """返回每个 segment 的时长（秒）"""
    if len(states) == 0:
        return np.array([])
    boundaries = np.where(np.diff(states, prepend=states[0] + 1) != 0)[0]
    durations = np.diff(np.append(boundaries, len(states))) / fs
    return durations


def compute_segmentation_quality(recs, n_states, fs):
    all_dur = np.concatenate([segment_durations(r["states"], fs) for r in recs])
    total_min = sum(len(r["states"]) for r in recs) / fs / 60.0
    n_segments = len(all_dur)

    # occupancy
    counts = np.zeros(n_states, dtype=int)
    for r in recs:
        counts += np.bincount(r["states"], minlength=n_states)
    occupancy = counts / counts.sum()
    occupancy_entropy = -np.sum(occupancy[occupancy > 0] * np.log(occupancy[occupancy > 0]))

    # self-transition
    trans = np.zeros((n_states, n_states), dtype=float)
    for r in recs:
        s = r["states"]
        if len(s) > 1:
            trans += np.eye(n_states)[s[:-1], :].T @ np.eye(n_states)[s[1:], :]
    trans_sum = trans.sum(axis=1, keepdims=True)
    trans_norm = np.divide(trans, trans_sum, out=np.zeros_like(trans), where=trans_sum > 0)
    self_ratio = np.trace(trans_norm) / max(np.trace(trans_norm) + 1e-12, 1e-12)
    if trans_norm.sum() > 0:
        self_ratio = np.trace(trans_norm) / trans_norm.sum()

    return {
        "n_segments_total": int(n_segments),
        "segments_per_minute": float(n_segments / total_min),
        "duration_mean_s": float(all_dur.mean()),
        "duration_median_s": float(np.median(all_dur)),
        "duration_std_s": float(all_dur.std()),
        "duration_min_s": float(all_dur.min()),
        "duration_max_s": float(all_dur.max()),
        "duration_p25_s": float(np.percentile(all_dur, 25)),
        "duration_p75_s": float(np.percentile(all_dur, 75)),
        "duration_p90_s": float(np.percentile(all_dur, 90)),
        "duration_p95_s": float(np.percentile(all_dur, 95)),
        "fraction_lt_0.1s": float((all_dur < 0.1).mean()),
        "fraction_lt_0.2s": float((all_dur < 0.2).mean()),
        "fraction_lt_0.5s": float((all_dur < 0.5).mean()),
        "state_occupancy_entropy": float(occupancy_entropy),
        "self_transition_ratio": float(self_ratio),
    }, all_dur, occupancy, trans_norm


def compute_state_consistency(recs, n_states, max_samples=50000):
    latents = [r["latent"] for r in recs]
    states = [r["states"] for r in recs]
    Z = np.concatenate(latents, axis=0)
    S = np.concatenate(states, axis=0)

    # per-state stats
    centroids = np.zeros((n_states, Z.shape[1]))
    covs = []
    within_vars = []
    state_ns = []
    for k in range(n_states):
        zk = Z[S == k]
        state_ns.append(len(zk))
        if len(zk) > 0:
            centroids[k] = zk.mean(axis=0)
            cov = np.cov(zk, rowvar=False)
            if zk.shape[1] == 1:
                cov = np.array([[cov]])
            covs.append(cov)
            within_vars.append(np.trace(cov) / zk.shape[1])
        else:
            covs.append(np.eye(Z.shape[1]) * 0)
            within_vars.append(np.nan)

    valid = np.array([len(Z[S == k]) > 0 for k in range(n_states)])
    valid_centroids = centroids[valid]
    between_dists = cdist(valid_centroids, valid_centroids, metric="euclidean")
    between_mean = between_dists[np.triu_indices_from(between_dists, k=1)].mean()

    # silhouette on subsample
    if len(Z) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Z), size=max_samples, replace=False)
        Zs, Ss = Z[idx], S[idx]
    else:
        Zs, Ss = Z, S
    sil = silhouette_score(Zs, Ss) if len(np.unique(Ss)) > 1 else np.nan

    return {
        "within_state_variance_mean": float(np.nanmean(within_vars)),
        "between_state_centroid_distance_mean": float(between_mean),
        "silhouette_score": float(sil),
        "state_n_samples": {int(k): int(n) for k, n in enumerate(state_ns)},
    }, centroids, valid


def compute_cross_recording_consistency(recs, n_states):
    occs = []
    stems = []
    for r in recs:
        occ = np.bincount(r["states"], minlength=n_states) / len(r["states"])
        occs.append(occ)
        stems.append(r["stem"])
    occs = np.array(occs)

    n = len(stems)
    js_mat = np.zeros((n, n))
    corr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            js_mat[i, j] = jensenshannon(occs[i], occs[j])
            corr_mat[i, j] = np.corrcoef(occs[i], occs[j])[0, 1]

    return {
        "occupancy_js_mean": float(js_mat[np.triu_indices_from(js_mat, k=1)].mean()),
        "occupancy_correlation_mean": float(corr_mat[np.triu_indices_from(corr_mat, k=1)].mean()),
        "recording_names": stems,
    }, occs, js_mat, corr_mat


def compute_temporal_structure(recs, n_states, fs):
    trans = np.zeros((n_states, n_states), dtype=float)
    for r in recs:
        s = r["states"]
        if len(s) > 1:
            for a, b in zip(s[:-1], s[1:]):
                trans[a, b] += 1
    row_sum = trans.sum(axis=1, keepdims=True)
    P = np.divide(trans, row_sum, out=np.zeros_like(trans), where=row_sum > 0)

    # transition entropy
    flat = P[P > 0]
    trans_entropy = float(-np.sum(flat * np.log(flat)))

    # entropy rate (stationary approximation)
    try:
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.argmax(np.real(eigvals))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        entropy_rate = float(-np.sum(stationary[:, None] * P * np.log(P + 1e-12)))
    except Exception:
        stationary = np.ones(n_states) / n_states
        entropy_rate = np.nan

    # autocorrelation of state sequence at multiple lags
    def state_autocorr(s, max_lag):
        s_centered = s - s.mean()
        c0 = np.dot(s_centered, s_centered) / len(s)
        ac = []
        for lag in range(1, max_lag + 1):
            c = np.dot(s_centered[:-lag], s_centered[lag:]) / len(s)
            ac.append(c / (c0 + 1e-12))
        return np.array(ac)

    max_lag = min(100, min(len(r["states"]) for r in recs) // 2)
    acs = [state_autocorr(r["states"], max_lag) for r in recs]
    autocorr = np.array(acs).mean(axis=0)
    lags = np.arange(1, max_lag + 1) / fs

    return {
        "transition_entropy": trans_entropy,
        "entropy_rate": entropy_rate,
        "autocorr_lags_s": lags.tolist(),
        "autocorr_values": autocorr.tolist(),
    }, P, stationary, lags, autocorr


def plot_results(exp_dir, all_dur, occupancy, trans_norm, stems, occs, js_mat, corr_mat, lags, autocorr, centroids, valid):
    out = exp_dir / "eval_hsmm_figures"
    out.mkdir(exist_ok=True)

    # 1. duration distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_dur, bins=100, color="steelblue", edgecolor="white")
    ax.axvline(np.median(all_dur), color="red", linestyle="--", label=f"median={np.median(all_dur):.2f}s")
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("count")
    ax.set_title("HSMM Segment Duration Distribution")
    ax.legend()
    ax.set_xlim(0, min(5, all_dur.max()))
    fig.tight_layout()
    fig.savefig(out / "segment_duration_distribution.png", dpi=150)
    plt.close(fig)

    # 2. occupancy
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(occupancy))
    ax.bar(x, occupancy, color="steelblue")
    ax.set_xlabel("state id")
    ax.set_ylabel("occupancy")
    ax.set_title("State Occupancy Distribution")
    fig.tight_layout()
    fig.savefig(out / "state_occupancy.png", dpi=150)
    plt.close(fig)

    # 3. transition matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(trans_norm, cmap="viridis", ax=ax, cbar_kws={"label": "transition probability"})
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    ax.set_title("HSMM Transition Matrix")
    fig.tight_layout()
    fig.savefig(out / "transition_matrix.png", dpi=150)
    plt.close(fig)

    # 4. recording similarity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(js_mat, xticklabels=stems, yticklabels=stems, cmap="rocket_r", ax=axes[0], annot=True, fmt=".2f")
    axes[0].set_title("Occupancy JS Divergence")
    sns.heatmap(corr_mat, xticklabels=stems, yticklabels=stems, cmap="vlag", center=0, ax=axes[1], annot=True, fmt=".2f")
    axes[1].set_title("Occupancy Pearson Correlation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(out / "recording_similarity_heatmap.png", dpi=150)
    plt.close(fig)

    # 5. autocorrelation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, autocorr, color="steelblue")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("State Sequence Autocorrelation")
    fig.tight_layout()
    fig.savefig(out / "state_autocorrelation.png", dpi=150)
    plt.close(fig)

    # 6. UMAP colored by state
    if valid.sum() > 1:
        Z = np.concatenate([r["latent"] for r in recs], axis=0)
        S = np.concatenate([r["states"] for r in recs], axis=0)
        if len(Z) > 50000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Z), 50000, replace=False)
            Z, S = Z[idx], S[idx]
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=S, cmap="tab20", s=1, alpha=0.5)
        ax.set_title("UMAP of Latent Colored by HSMM State")
        plt.colorbar(scatter, ax=ax, label="state")
        fig.tight_layout()
        fig.savefig(out / "state_umap.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate HSMM results statistically")
    parser.add_argument("--exp", type=str, default=None, help="Experiment number, e.g. 008. Default: latest")
    args = parser.parse_args()

    cfg = load_config()
    output_root = Path(cfg["project"]["output_dir"])
    if args.exp:
        exp_dir = output_root / f"exp_{args.exp.zfill(3)}"
    else:
        exp_dir = find_latest_exp(output_root)
    print(f"评估目录: {exp_dir}")

    fs = cfg["preprocess"]["target_fs"]
    n_states = cfg["hmm"]["n_states"]

    # load all recordings
    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))
    recs = []
    for hf in hsmm_files:
        stem = hf.name.replace("_hsmm.npz", "")
        recs.append(load_pair(exp_dir, stem))
    print(f"加载 {len(recs)} 个 recording")

    # metrics
    seg_metrics, all_dur, occupancy, trans_norm = compute_segmentation_quality(recs, n_states, fs)
    cons_metrics, centroids, valid = compute_state_consistency(recs, n_states)
    cross_metrics, occs, js_mat, corr_mat = compute_cross_recording_consistency(recs, n_states)
    temp_metrics, P, stationary, lags, autocorr = compute_temporal_structure(recs, n_states, fs)

    report = {
        "experiment": exp_dir.name,
        "n_recordings": len(recs),
        "n_states": n_states,
        "fs_hz": fs,
        "segmentation": seg_metrics,
        "state_consistency": cons_metrics,
        "cross_recording": cross_metrics,
        "temporal_structure": temp_metrics,
    }

    # save
    with open(exp_dir / "eval_hsmm_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"报告已保存: {exp_dir / 'eval_hsmm_report.json'}")

    df_occ = pd.DataFrame(occs, columns=[f"state_{i}" for i in range(n_states)], index=cross_metrics["recording_names"])
    df_occ.to_csv(exp_dir / "eval_hsmm_occupancy.csv")

    # plots
    plot_results(exp_dir, all_dur, occupancy, trans_norm,
                 cross_metrics["recording_names"], occs, js_mat, corr_mat,
                 lags, autocorr, centroids, valid)
    print(f"图像已保存: {exp_dir / 'eval_hsmm_figures'}")


if __name__ == "__main__":
    main()