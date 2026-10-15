"""HSMM state 自动合并提议。

根据 state 在 latent 空间中的分布相似度做层次聚类，
输出候选 merge_groups，供人工确认。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from utils import load_config, output_dir, latest_experiment_dir


def resolve_exp_dir(args_exp):
    cfg = load_config()
    out = output_dir(cfg)
    if args_exp:
        p = Path(args_exp)
        return p if p.is_absolute() else out / p
    return latest_experiment_dir(cfg)


def load_hsmm_data(exp_dir, rec_filter=None):
    """加载 *_hsmm.npz 和对应 *_latent.npz。"""
    exp_dir = Path(exp_dir)
    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))

    if rec_filter is not None:
        rec_set = set(rec_filter)
        hsmm_files = [p for p in hsmm_files
                      if p.stem.replace("_hsmm", "") in rec_set]

    if not hsmm_files:
        raise FileNotFoundError(f"{exp_dir} 中没有匹配的 *_hsmm.npz")

    data = []
    for p in hsmm_files:
        rec = p.stem.replace("_hsmm", "")
        hsmm = np.load(p)
        latent = np.load(exp_dir / f"{rec}_latent.npz")

        states = hsmm["states"]
        z = latent["downstream"]
        time = latent["time"]

        # 对齐长度（latent 可能比 state 短 due to warmup）
        min_len = min(len(states), len(z))
        data.append({
            "recording": rec,
            "states": states[:min_len],
            "z": z[:min_len],
            "time": time[:min_len],
        })
    return data


def compute_state_centroids(data, min_samples=10):
    """计算每个 state 的 latent 均值和协方差。"""
    all_states = np.concatenate([d["states"] for d in data])
    all_z = np.concatenate([d["z"] for d in data])

    n_states = int(all_states.max()) + 1
    centroids = []
    covs = []
    counts = []

    for s in range(n_states):
        mask = all_states == s
        count = mask.sum()
        counts.append(int(count))

        if count >= min_samples:
            zs = all_z[mask]
            centroids.append(zs.mean(axis=0))
            covs.append(np.cov(zs, rowvar=False))
        else:
            # 样本太少，给一个很大的值，避免被错误合并
            centroids.append(np.full(all_z.shape[1], 1e6))
            covs.append(np.eye(all_z.shape[1]) * 1e6)

    return np.array(centroids), covs, np.array(counts), n_states


def bhattacharyya_distance(mu1, mu2, cov1, cov2):
    """两个高斯分布之间的 Bhattacharyya 距离。"""
    cov = (cov1 + cov2) / 2.0
    diff = mu1 - mu2

    try:
        inv_cov = np.linalg.inv(cov)
        term1 = 0.125 * diff @ inv_cov @ diff
        sign, logdet = np.linalg.slogdet(cov)
        sign1, logdet1 = np.linalg.slogdet(cov1)
        sign2, logdet2 = np.linalg.slogdet(cov2)
        if sign <= 0 or sign1 <= 0 or sign2 <= 0:
            return np.inf
        term2 = 0.5 * (logdet - 0.5 * (logdet1 + logdet2))
        return float(term1 + term2)
    except np.linalg.LinAlgError:
        return np.inf


def compute_distance_matrix(centroids, covs, metric="euclidean"):
    """计算 state 两两距离矩阵。"""
    n = len(centroids)

    if metric == "euclidean":
        dists = squareform(pdist(centroids, metric="euclidean"))
    elif metric == "bhattacharyya":
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = bhattacharyya_distance(centroids[i], centroids[j],
                                           covs[i], covs[j])
                dists[i, j] = d
                dists[j, i] = d
    else:
        raise ValueError(f"未知 metric: {metric}")

    # 把 inf/nan 替换为最大值
    max_d = np.nanmax(dists[np.isfinite(dists)])
    dists[~np.isfinite(dists)] = max_d * 2
    return dists


def build_merge_groups(cluster_labels):
    """从聚类标签生成 merge_groups。"""
    groups = {}
    for state, label in enumerate(cluster_labels):
        groups.setdefault(int(label), []).append(int(state))
    return {f"group_{k:02d}": v for k, v in groups.items()}


def main():
    parser = argparse.ArgumentParser(description="HSMM state 自动合并提议")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验目录，默认最新 exp")
    parser.add_argument("--recs", type=str, default=None,
                        help="逗号分隔的 recording 白名单，默认全量")
    parser.add_argument("--metric", type=str, default="euclidean",
                        choices=["euclidean", "bhattacharyya"],
                        help="state 间距离度量")
    parser.add_argument("--method", type=str, default="ward",
                        choices=["ward", "average", "complete", "single"],
                        help="层次聚类方法")
    parser.add_argument("--max_k", type=int, default=25,
                        help="最大合并后类别数")
    parser.add_argument("--min_samples", type=int, default=10,
                        help="计算协方差所需最小样本数")
    args = parser.parse_args()

    exp_dir = resolve_exp_dir(args.exp)
    rec_filter = [r.strip() for r in args.recs.split(",") if r.strip()] if args.recs else None

    print(f"实验目录: {exp_dir}")
    if rec_filter:
        print(f"处理 recordings: {rec_filter}")
    else:
        print("处理全部 recordings")

    # 加载数据
    data = load_hsmm_data(exp_dir, rec_filter=rec_filter)
    rec_names = "_".join(sorted(d["recording"] for d in data))
    if len(rec_names) > 100:
        rec_names = f"{len(data)}recs"

    # 计算 state 分布
    centroids, covs, counts, n_states = compute_state_centroids(data, min_samples=args.min_samples)
    print(f"state 数: {n_states}")
    print(f"每个 state 样本数: {counts}")

    # 距离矩阵
    print(f"计算 {args.metric} 距离矩阵...")
    dists = compute_distance_matrix(centroids, covs, metric=args.metric)

    # 层次聚类
    # linkage 需要 condensed distance matrix
    cond_dists = squareform(dists, checks=False)
    Z = linkage(cond_dists, method=args.method)

    # 输出目录
    out_dir = exp_dir / "merge_proposals"
    out_dir.mkdir(exist_ok=True)

    # 生成不同 K 的合并提议
    proposals = {}
    for k in range(args.max_k, 1, -5):
        if k >= n_states:
            continue
        labels = fcluster(Z, t=k, criterion="maxclust")
        proposals[f"k{k}"] = build_merge_groups(labels)

    # 保存 JSON
    rec_tag = rec_names if rec_filter else "all"
    out_json = out_dir / f"merge_proposal_{rec_tag}_{args.metric}_{args.method}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "recordings": [d["recording"] for d in data],
            "n_states": int(n_states),
            "metric": args.metric,
            "method": args.method,
            "state_counts": counts.tolist(),
            "proposals": proposals,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n合并提议保存: {out_json}")

    # 画 dendrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f"State hierarchical clustering ({args.metric}, {args.method})\n{rec_tag}")
    ax.set_xlabel("state")
    ax.set_ylabel("distance")
    fig.tight_layout()
    out_png = out_dir / f"dendrogram_{rec_tag}_{args.metric}_{args.method}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"树状图保存: {out_png}")

    # 打印最相似的 state 对
    print("\n最相似的 10 对 state（建议优先检查）：")
    triu_idx = np.triu_indices(n_states, k=1)
    sorted_idx = np.argsort(dists[triu_idx])
    for rank, idx in enumerate(sorted_idx[:10]):
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        print(f"  {rank+1:2d}. state {i:2d} ↔ state {j:2d}  "
              f"distance={dists[i,j]:.3f}  "
              f"counts=({counts[i]}, {counts[j]})")


if __name__ == "__main__":
    main()