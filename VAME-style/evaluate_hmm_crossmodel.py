"""HSMM 片段级轨迹评估。
比较两个 HSMM 结果，基于 latent 轨迹的动态模式而非单帧聚类。
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import entropy

try:
    from tslearn.metrics import dtw
    from tslearn.barycenters import dtw_barycenter_averaging
except ImportError as e:
    raise ImportError("请先安装 tslearn：pip install tslearn") from e

try:
    import umap
except ImportError as e:
    raise ImportError("请先安装 umap-learn：pip install umap-learn") from e


# ===================== 用户修改这里 =====================
EXP_A = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\Batch3\exp_008")
EXP_B = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\Batch_SplitG1\exp_002")
OUT_DIR = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\hsmm_segment_compare")
FS = 25.0                       # 采样率
TARGET_LEN = 25                 # 片段重采样目标长度（帧），默认 1 秒
MAX_SEGMENTS_PER_STATE = 150    # 每个状态最多用多少片段，控制速度
MAX_PROTOTYPE_LEN = 100         # 原型最大长度，避免 DBA 过慢
# ========================================================


warnings.filterwarnings("ignore")


def load_hsmm_results(exp_dir):
    """读取 *_hsmm.npz 和对应 *_latent.npz。"""
    hsmm_files = sorted(exp_dir.glob("*_hsmm.npz"))
    if not hsmm_files:
        raise FileNotFoundError(f"{exp_dir} 下没有找到 *_hsmm.npz 文件")

    recs = []
    for f in hsmm_files:
        rec_name = f.stem.replace("_hsmm", "")
        latent_file = f.parent / f"{rec_name}_latent.npz"
        if not latent_file.exists():
            raise FileNotFoundError(f"找不到 latent 文件: {latent_file}")

        hsmm_data = np.load(f)
        latent_data = np.load(latent_file)

        states = hsmm_data["states"].astype(np.int64)
        downstream = latent_data["downstream"].astype(np.float32)

        min_len = min(len(states), len(downstream))
        states = states[:min_len]
        downstream = downstream[:min_len]

        recs.append({
            "name": rec_name,
            "states": states,
            "downstream": downstream,
        })
    return recs


def extract_segments(states):
    """把状态序列切分为 (state, start, duration)。"""
    if len(states) == 0:
        return []
    segments = []
    s = 0
    cur = states[0]
    for i in range(1, len(states)):
        if states[i] != cur:
            segments.append((int(cur), s, i - s))
            cur = states[i]
            s = i
    segments.append((int(cur), s, len(states) - s))
    return segments


def resample_segment(seg, target_len):
    """把片段重采样到固定长度。"""
    seg = np.asarray(seg)
    if len(seg) == target_len:
        return seg
    if len(seg) < 2:
        return np.repeat(seg, target_len, axis=0)[:target_len]
    x_old = np.linspace(0, 1, len(seg))
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, seg, axis=0, kind="linear", fill_value="extrapolate")
    return f(x_new)


def compute_state_prototype(segments):
    """用 DBA 计算某状态的典型轨迹原型。"""
    if len(segments) == 0:
        return None

    # 限制片段数量
    if len(segments) > MAX_SEGMENTS_PER_STATE:
        idx = np.random.choice(len(segments), MAX_SEGMENTS_PER_STATE, replace=False)
        segments = [segments[i] for i in idx]

    # 限制原型长度
    target_len = min(max(len(s) for s in segments), MAX_PROTOTYPE_LEN)
    segments_rs = [resample_segment(s, target_len) for s in segments]

    prototype = dtw_barycenter_averaging(
        segments_rs,
        max_iter=10,
        init_barycenter=segments_rs[0],
        verbose=False,
    )
    return prototype


def compute_intra_state_dtw(segments, prototype):
    """计算同状态内片段到原型的平均 DTW 距离。"""
    if prototype is None or len(segments) == 0:
        return np.nan

    proto_len = len(prototype)
    distances = []
    for seg in segments:
        if len(seg) < 2:
            continue
        seg_rs = resample_segment(seg, proto_len)
        d = dtw(seg_rs, prototype)
        distances.append(d)

    return float(np.mean(distances)) if distances else np.nan


def compute_inter_state_dtw(prototypes):
    """计算不同状态原型之间的 DTW 距离矩阵。"""
    valid = {k: p for k, p in prototypes.items() if p is not None}
    if len(valid) < 2:
        return np.array([]), []

    states = sorted(valid.keys())
    target_len = min(max(len(valid[k]) for k in states), MAX_PROTOTYPE_LEN)
    protos = [resample_segment(valid[k], target_len) for k in states]

    n = len(states)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(protos[i], protos[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat, states


def compute_segment_features(segments, target_len=TARGET_LEN):
    """把片段展平为固定长度向量，用于 UMAP。"""
    feats = []
    labels = []
    for st, segs in segments.items():
        for seg in segs:
            if len(seg) < 2:
                continue
            seg_rs = resample_segment(seg, target_len)
            feats.append(seg_rs.flatten())
            labels.append(st)
    return np.array(feats), np.array(labels)


def evaluate_model(exp_dir):
    """评估单个 HSMM 结果。"""
    print(f"\n评估: {exp_dir}")
    recs = load_hsmm_results(exp_dir)

    n_states = int(np.max([r["states"].max() for r in recs]) + 1)

    # 按状态收集片段
    state_segments = {k: [] for k in range(n_states)}
    for r in recs:
        segs = extract_segments(r["states"])
        for st, start, dur in segs:
            if dur >= 2:  # 过滤过短片段
                state_segments[st].append(r["downstream"][start:start + dur])

    active_states = [k for k, segs in state_segments.items() if len(segs) > 0]
    n_active = len(active_states)
    print(f"  总状态数: {n_states}, 活跃状态数: {n_active}")

    # 计算原型
    prototypes = {}
    for st in active_states:
        prototypes[st] = compute_state_prototype(state_segments[st])

    # 同状态内距离
    intra_dists = {}
    for st in active_states:
        intra_dists[st] = compute_intra_state_dtw(state_segments[st], prototypes[st])

    # 原型间距离
    inter_dist_mat, inter_states = compute_inter_state_dtw(prototypes)

    # 综合指标
    intra_values = [v for v in intra_dists.values() if np.isfinite(v)]
    mean_intra = float(np.mean(intra_values)) if intra_values else np.nan

    mean_inter = np.nan
    if inter_dist_mat.size > 0:
        triu = np.triu_indices_from(inter_dist_mat, k=1)
        mean_inter = float(np.mean(inter_dist_mat[triu]))

    separability = mean_inter / (mean_intra + 1e-12) if np.isfinite(mean_inter) and np.isfinite(mean_intra) else np.nan

    # UMAP 片段特征
    # 片段特征降维：优先 UMAP，失败则用 PCA
    feats, labels = compute_segment_features(state_segments)
    segment_emb = None
    reducer_name = None
    if len(feats) > 10 and len(np.unique(labels)) > 1:
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
            segment_emb = reducer.fit_transform(feats)
            reducer_name = "umap"
        except Exception as e:
            print(f"  UMAP 失败 ({e})，回退到 PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            segment_emb = reducer.fit_transform(feats)
            reducer_name = "pca"

    return {
        "exp_dir": str(exp_dir),
        "n_states": n_states,
        "n_active_states": n_active,
        "mean_intra_dtw": mean_intra,
        "mean_inter_dtw": mean_inter,
        "separability_ratio": separability,
        "per_state_intra_dtw": {int(k): float(v) for k, v in intra_dists.items()},
        "prototype_inter_dtw_matrix": inter_dist_mat.tolist(),
        "prototype_states": inter_states,
        "segment_features": feats.tolist(),
        "segment_labels": labels.tolist(),
        "segment_embedding": segment_emb.tolist() if segment_emb is not None else None,
        "reducer": reducer_name,
    }


def plot_comparison(result_a, result_b, out_dir):
    """绘制对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 指标对比
    ax = axes[0, 0]
    metrics = ["mean_intra_dtw", "mean_inter_dtw", "separability_ratio"]
    names = [Path(result_a["exp_dir"]).name, Path(result_b["exp_dir"]).name]
    x = np.arange(len(metrics))
    width = 0.35
    vals_a = [result_a[m] for m in metrics]
    vals_b = [result_b[m] for m in metrics]
    ax.bar(x - width/2, vals_a, width, label=names[0])
    ax.bar(x + width/2, vals_b, width, label=names[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["Intra-state DTW", "Inter-state DTW", "Separability"])
    ax.legend()
    ax.set_title("Segment-level DTW metrics")

    # 2. 每个状态的内距
    ax = axes[0, 1]
    common_states = sorted(set(result_a["per_state_intra_dtw"].keys()) & set(result_b["per_state_intra_dtw"].keys()))
    if common_states:
        ax.plot(common_states, [result_a["per_state_intra_dtw"][s] for s in common_states], "o-", label=names[0])
        ax.plot(common_states, [result_b["per_state_intra_dtw"][s] for s in common_states], "s-", label=names[1])
        ax.set_xlabel("state")
        ax.set_ylabel("intra-state DTW")
        ax.legend()
        ax.set_title("Per-state compactness")

    # 3. UMAP A
    ax = axes[1, 0]
    emb_a = result_a.get("umap_embedding")
    if emb_a is not None:
        emb_a = np.array(emb_a)
        labels_a = np.array(result_a["umap_labels"])
        scatter = ax.scatter(emb_a[:, 0], emb_a[:, 1], c=labels_a, cmap="tab20", s=5, alpha=0.6)
        ax.set_title(f"UMAP segments: {names[0]}")
        plt.colorbar(scatter, ax=ax)

    # 4. UMAP B
    ax = axes[1, 1]
    emb_b = result_b.get("umap_embedding")
    if emb_b is not None:
        emb_b = np.array(emb_b)
        labels_b = np.array(result_b["umap_labels"])
        scatter = ax.scatter(emb_b[:, 0], emb_b[:, 1], c=labels_b, cmap="tab20", s=5, alpha=0.6)
        ax.set_title(f"UMAP segments: {names[1]}")
        plt.colorbar(scatter, ax=ax)

    plt.tight_layout()
    fig.savefig(out_dir / "segment_comparison.png", dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result_a = evaluate_model(EXP_A)
    result_b = evaluate_model(EXP_B)

    summary = {
        "model_a": result_a,
        "model_b": result_b,
    }

    # 保存 JSON
    with open(OUT_DIR / "segment_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 保存简洁对比表
    import csv
    rows = [
        {"metric": "n_active_states", "model_a": result_a["n_active_states"], "model_b": result_b["n_active_states"]},
        {"metric": "mean_intra_dtw", "model_a": result_a["mean_intra_dtw"], "model_b": result_b["mean_intra_dtw"]},
        {"metric": "mean_inter_dtw", "model_a": result_a["mean_inter_dtw"], "model_b": result_b["mean_inter_dtw"]},
        {"metric": "separability_ratio", "model_a": result_a["separability_ratio"], "model_b": result_b["separability_ratio"]},
    ]
    with open(OUT_DIR / "segment_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "model_a", "model_b"])
        writer.writeheader()
        writer.writerows(rows)

    plot_comparison(result_a, result_b, OUT_DIR)

    print(f"\n{'='*60}")
    print("片段级评估结果")
    print(f"{'='*60}")
    print(f"{'metric':<25} {Path(result_a['exp_dir']).name:<15} {Path(result_b['exp_dir']).name:<15}")
    print("-" * 60)
    for row in rows:
        print(f"{row['metric']:<25} {row['model_a']:<15.4f} {row['model_b']:<15.4f}")

    print(f"\n结果已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()