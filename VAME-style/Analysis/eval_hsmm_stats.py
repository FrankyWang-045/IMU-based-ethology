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
    # 保留原始有效帧：interp_mask == 0
    mask = hsmm["interp_mask"] == 0
    return {
        "stem": stem,
        "states": hsmm["states"][mask],
        "time": hsmm["time"][mask],
        "latent": latent["downstream"][mask],
    }

def segment_durations(states, fs):
    if len(states) == 0:
        return np.array([])
    boundaries = np.where(np.diff(states, prepend=states[0] + 1) != 0)[0]
    durations = np.diff(np.append(boundaries, len(states))) / fs
    return durations


def compute_raw_occupancy(recs, n_states):
    counts = np.zeros(n_states, dtype=int)
    for r in recs:
        counts += np.bincount(r["states"], minlength=n_states)
    return counts / counts.sum()

def remap_to_active_states(recs, occupancy):
    """把 dead state 排除，返回新的 recs 和 active state 信息"""
    active_old = np.where(occupancy > 0)[0]
    old2new = {int(old): new for new, old in enumerate(active_old)}
    new2old = {new: int(old) for new, old in enumerate(active_old)}

    new_recs = []
    for r in recs:
        mapped = np.array(
            [old2new.get(int(s), -1) for s in r["states"]],
            dtype=np.int64,
        )
        keep = mapped >= 0
        new_recs.append({
            "stem": r["stem"],
            "states": mapped[keep],
            "time": r["time"][keep],
            "latent": r["latent"][keep],
        })
    return new_recs, active_old, new2old


def compute_segmentation_quality(recs, n_active, fs):
    all_dur = np.concatenate([segment_durations(r["states"], fs) for r in recs])
    total_min = sum(len(r["states"]) for r in recs) / fs / 60.0
    n_segments = len(all_dur)

    counts = np.zeros(n_active, dtype=int)
    for r in recs:
        counts += np.bincount(r["states"], minlength=n_active)
    occupancy = counts / counts.sum()
    occupancy_entropy = -np.sum(occupancy[occupancy > 0] * np.log(occupancy[occupancy > 0]))

    trans = np.zeros((n_active, n_active), dtype=float)
    for r in recs:
        s = r["states"]
        if len(s) > 1:
            trans += np.eye(n_active)[s[:-1], :].T @ np.eye(n_active)[s[1:], :]
    trans_sum = trans.sum(axis=1, keepdims=True)
    trans_norm = np.divide(trans, trans_sum, out=np.zeros_like(trans), where=trans_sum > 0)
    self_ratio = np.trace(trans_norm) / max(trans_norm.sum(), 1e-12)

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


def compute_state_consistency(recs, n_active, max_samples=50000):
    latents = [r["latent"] for r in recs]
    states = [r["states"] for r in recs]
    Z = np.concatenate(latents, axis=0)
    S = np.concatenate(states, axis=0)

    centroids = np.zeros((n_active, Z.shape[1]))
    within_vars = []
    state_ns = []
    for k in range(n_active):
        zk = Z[S == k]
        state_ns.append(len(zk))
        if len(zk) > 1:
            centroids[k] = zk.mean(axis=0)
            cov = np.cov(zk, rowvar=False)
            if zk.shape[1] == 1:
                cov = np.array([[cov]])
            within_vars.append(np.trace(cov) / zk.shape[1])
        elif len(zk) == 1:
            centroids[k] = zk[0]
            within_vars.append(0.0)
        else:
            within_vars.append(np.nan)

    valid = np.array(state_ns) > 0
    valid_centroids = centroids[valid]
    if valid_centroids.shape[0] > 1:
        between_dists = cdist(valid_centroids, valid_centroids, metric="euclidean")
        between_mean = between_dists[np.triu_indices_from(between_dists, k=1)].mean()
    else:
        between_mean = np.nan

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


def compute_cross_recording_consistency(recs, n_active):
    occs = []
    stems = []
    for r in recs:
        occ = np.bincount(r["states"], minlength=n_active) / len(r["states"])
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


def compute_temporal_structure(recs, n_active, fs):
    trans = np.zeros((n_active, n_active), dtype=float)
    for r in recs:
        s = r["states"]
        if len(s) > 1:
            for a, b in zip(s[:-1], s[1:]):
                trans[a, b] += 1
    row_sum = trans.sum(axis=1, keepdims=True)
    P = np.divide(trans, row_sum, out=np.zeros_like(trans), where=row_sum > 0)

    flat = P[P > 0]
    trans_entropy = float(-np.sum(flat * np.log(flat)))

    try:
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.argmax(np.real(eigvals))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        entropy_rate = float(-np.sum(stationary[:, None] * P * np.log(P + 1e-12)))
    except Exception:
        stationary = np.ones(n_active) / n_active
        entropy_rate = np.nan

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


def plot_results(exp_dir, recs, all_dur, occupancy, trans_norm, stems, occs, js_mat, corr_mat, lags, autocorr, centroids, valid, new2old):
    out = exp_dir / "eval_hsmm_figures"
    out.mkdir(exist_ok=True)

    # 1. duration distribution：线性 + 半对数
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.hist(all_dur, bins=100, color="steelblue", edgecolor="white")
    ax.axvline(np.median(all_dur), color="red", linestyle="--", label=f"median={np.median(all_dur):.2f}s")
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("count")
    ax.set_title("Segment Duration (linear)")
    ax.legend()
    ax.set_xlim(0, min(5, all_dur.max()))

    ax = axes[1]
    ax.hist(all_dur, bins=100, color="steelblue", edgecolor="white")
    ax.set_yscale("log")
    ax.axvline(np.median(all_dur), color="red", linestyle="--", label=f"median={np.median(all_dur):.2f}s")
    ax.set_xlabel("segment duration (s)")
    ax.set_ylabel("count (log)")
    ax.set_title("Segment Duration (log-y)")
    ax.legend()
    ax.set_xlim(0, min(5, all_dur.max()))

    fig.tight_layout()
    fig.savefig(out / "segment_duration_distribution.png", dpi=150)
    plt.close(fig)

    # 2. occupancy (x轴显示原始 state id)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = [new2old[i] for i in range(len(occupancy))]
    ax.bar(x, occupancy, color="steelblue")
    ax.set_xlabel("original state id")
    ax.set_ylabel("occupancy")
    ax.set_title("State Occupancy Distribution (dead states excluded)")
    fig.tight_layout()
    fig.savefig(out / "state_occupancy.png", dpi=150)
    plt.close(fig)

    # 3. transition matrix：原始 + off-diagonal 归一化
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 左图：原始转移矩阵
    ax = axes[0]
    sns.heatmap(trans_norm, cmap="viridis", ax=ax, cbar_kws={"label": "probability"})
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    ax.set_title("Raw Transition Matrix")

    # 右图：off-diagonal 归一化（去掉自转移后重新按行归一化）
    ax = axes[1]
    P_off = trans_norm.copy()
    np.fill_diagonal(P_off, 0.0)
    row_sum = P_off.sum(axis=1, keepdims=True)
    P_off_norm = np.divide(P_off, row_sum, out=np.zeros_like(P_off), where=row_sum > 0)
    sns.heatmap(P_off_norm, cmap="magma", ax=ax, cbar_kws={"label": "off-diagonal probability"})
    ax.set_xlabel("next state")
    ax.set_ylabel("current state")
    ax.set_title("Transition Matrix (self-transition excluded, row-normalized)")

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

    # 6. UMAP colored by original state id
    if valid.sum() > 1:
        Z = np.concatenate([r["latent"] for r in recs], axis=0)
        S = np.concatenate([r["states"] for r in recs], axis=0)
        S_orig = np.array([new2old[s] for s in S])
        if len(Z) > 50000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Z), 50000, replace=False)
            Z, S_orig = Z[idx], S_orig[idx]
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=S_orig, cmap="tab20", s=1, alpha=0.5)
        ax.set_title("UMAP of Latent Colored by HSMM State")
        plt.colorbar(scatter, ax=ax, label="original state id")
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
    n_states_total = cfg["hmm"]["n_states"]

    # 读取验证集文件列表
    val_files = set()
    val_json = exp_dir / "val_files.json"
    if val_json.exists():
        with open(val_json, "r", encoding="utf-8") as f:
            val_files = set(json.load(f))
        print(f"验证集文件: {sorted(val_files)}")

    # 加载所有 HSMM 结果（训练集 + 验证集）
    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))
    recs = []
    for hf in hsmm_files:
        stem = hf.name.replace("_hsmm.npz", "")
        recs.append(load_pair(exp_dir, stem))

    train_stems = [r["stem"] for r in recs if r["stem"] not in val_files]
    val_stems = [r["stem"] for r in recs if r["stem"] in val_files]
    print(f"加载 {len(recs)} 个 recording，训练 {len(train_stems)}，验证 {len(val_stems)}")

    # 先计算原始 occupancy，识别 dead states
    raw_occupancy = compute_raw_occupancy(recs, n_states_total)
    n_dead = int((raw_occupancy == 0).sum())
    n_active = n_states_total - n_dead
    print(f"总 state 数: {n_states_total}, active: {n_active}, dead: {n_dead}")

    # 排除 dead state 后重新映射
    recs_active, active_old, new2old = remap_to_active_states(recs, raw_occupancy)

    # metrics
    seg_metrics, all_dur, occupancy, trans_norm = compute_segmentation_quality(recs_active, n_active, fs)
    cons_metrics, centroids, valid = compute_state_consistency(recs_active, n_active)
    cross_metrics, occs, js_mat, corr_mat = compute_cross_recording_consistency(recs_active, n_active)
    temp_metrics, P, stationary, lags, autocorr = compute_temporal_structure(recs_active, n_active, fs)

    report = {
        "experiment": exp_dir.name,
        "n_recordings": len(recs),
        "train_recordings": train_stems,
        "val_recordings": val_stems,
        "n_states_total": n_states_total,
        "n_states_active": n_active,
        "n_dead_states": n_dead,
        "active_state_ids": active_old.tolist(),
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

    df_occ = pd.DataFrame(occs, columns=[f"state_{new2old[i]}" for i in range(n_active)], index=cross_metrics["recording_names"])
    df_occ.to_csv(exp_dir / "eval_hsmm_occupancy.csv")

    # plots
    plot_results(exp_dir, recs_active, all_dur, occupancy, trans_norm,
                 cross_metrics["recording_names"], occs, js_mat, corr_mat,
                 lags, autocorr, centroids, valid, new2old)
    print(f"图像已保存: {exp_dir / 'eval_hsmm_figures'}")


if __name__ == "__main__":
    main()