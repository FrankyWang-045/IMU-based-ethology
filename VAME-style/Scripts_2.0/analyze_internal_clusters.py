"""HSMM 指定 state 的 segment 级内部聚类分析（增强特征版）。

对 state 3, 5, 11, 13 的连续片段提取更丰富的 segment-level 特征：
- latent 统计量
- 6 轴信号统计量
- 6 轴动态特征（能量、主频、变化率）
- segment 时长与 latent 轨迹特征
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import umap.umap_ as _umap_mod

_orig_check_array = _umap_mod.check_array


def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)


_umap_mod.check_array = _check_array_compat

from utils import load_config, output_dir, latent_path, hmm_hsmm_path

TARGET_STATES = [3, 5, 11, 13]
K_RANGE = [2, 3, 4]
MIN_SEG_FRAMES = 20       # 最短 20 帧（25 Hz 下 0.8 s），保证动态特征有意义
N_PLOT_SEGS = 15
OUT_DIR = Path(
    r"F:\IMU_computational_ethology\Data\VAME-Style-Data\2.0smalltest\eval\internal_clustering"
)


def load_all_data(cfg):
    """加载所有 recording 的 latent、state、features。"""
    out = output_dir(cfg)
    latent_files = sorted(out.glob("*_latent.npz"))

    if not latent_files:
        raise FileNotFoundError(
            f"{out} 下没有 *_latent.npz 文件，请先运行 embed.py。"
        )

    zs, states_list, feats_list = [], [], []

    for latent_file in latent_files:
        rec = latent_file.stem.replace("_latent", "")
        latent = np.load(latent_file)
        hmm = np.load(hmm_hsmm_path(cfg, rec))
        npz_path = out / f"{rec}.npz"
        npz = np.load(npz_path)

        z = latent["downstream"]
        states = hmm["states"]
        feats = npz["features"]
        time_h = hmm["time"]
        time_f = npz["time"]

        idx = np.searchsorted(time_f, time_h)
        if not np.allclose(time_f[idx], time_h):
            raise ValueError(f"{rec} 时间对齐失败")
        feats_aligned = feats[idx]

        zs.append(z)
        states_list.append(states)
        feats_list.append(feats_aligned)

    z_all = np.concatenate(zs, axis=0)
    states_all = np.concatenate(states_list, axis=0)
    feats_all = np.concatenate(feats_list, axis=0)

    return z_all, states_all, feats_all


def extract_segments(z, states, feats, target_state, min_len=MIN_SEG_FRAMES):
    """提取目标 state 的连续片段。"""
    segments = []
    t = 0
    n = len(states)
    while t < n:
        if states[t] == target_state:
            s = t
            while t < n and states[t] == target_state:
                t += 1
            e = t
            if e - s >= min_len:
                segments.append(
                    {
                        "start": s,
                        "end": e,
                        "duration": e - s,
                        "z": z[s:e],
                        "feats": feats[s:e],
                    }
                )
        else:
            t += 1
    return segments


def dominant_frequency(x):
    """计算一维信号的主导频率（FFT 峰值对应的 bin）。"""
    if len(x) < 4:
        return 0
    fft = np.abs(np.fft.rfft(x - x.mean()))
    # 忽略直流分量
    if len(fft) > 1:
        return int(np.argmax(fft[1:]) + 1)
    return 0


def compute_segment_features(seg):
    """计算 segment 的增强特征向量。"""
    z = seg["z"]
    f = seg["feats"]

    # 1. latent 统计量 (12)
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)

    # 2. latent 轨迹特征 (12)
    z_start = z[0]
    z_end = z[-1]
    z_displacement = z_end - z_start
    z_path_length = np.abs(np.diff(z, axis=0)).sum(axis=0)

    # 3. 6 轴信号统计量 (18)
    f_mean = f.mean(axis=0)
    f_std = f.std(axis=0)
    f_range = f.max(axis=0) - f.min(axis=0)
    f_energy = (f ** 2).mean(axis=0)

    # 4. 6 轴动态特征 (18)
    f_diff = np.diff(f, axis=0)
    f_mad = np.abs(f_diff).mean(axis=0)  # mean absolute derivative
    f_fft_peak = np.array([dominant_frequency(f[:, i]) for i in range(f.shape[1])])

    # 5. 时长 (1)
    duration = np.array([seg["duration"]])

    return np.concatenate([
        z_mean, z_std,
        z_displacement, z_path_length,
        f_mean, f_std, f_range, f_energy,
        f_mad, f_fft_peak,
        duration,
    ])


def cluster_segments(segments, k_range):
    """对 segment 特征做聚类。"""
    if len(segments) < max(k_range) * 5:
        print(f"  segment 数量太少 ({len(segments)})，跳过聚类")
        return None, None, None

    X = np.stack([compute_segment_features(s) for s in segments])
    X_scaled = StandardScaler().fit_transform(X)

    best_k, best_sil, best_labels = None, -1, None
    results = []

    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append((k, labels, sil))
        print(f"  K={k}: silhouette={sil:.3f}")

        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    print(f"  最佳 K={best_k}, silhouette={best_sil:.3f}")

    return X_scaled, best_labels, results


def plot_umap(X, labels, target_state, k, out_dir):
    """画 segment 特征的 UMAP。"""
    if len(X) < 30:
        return

    emb = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for lab in np.unique(labels):
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=20, alpha=0.7, label=f"sub-{lab}")
    ax.legend()
    ax.set_title(f"state {target_state} segment clustering (K={k}, n={len(X)})")
    fig.tight_layout()
    fig.savefig(out_dir / f"state_{target_state:02d}_K{k}_umap.png", dpi=150)
    plt.close(fig)


def plot_cluster_signals(segments, labels, target_state, k, out_dir):
    """为每个 sub-cluster 画代表性 segment 的 raw 信号。"""
    axes_names = ["ax", "ay", "az", "wx", "wy", "wz"]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for lab in np.unique(labels):
        lab_segments = [seg for seg, lbl in zip(segments, labels) if lbl == lab]
        n = min(N_PLOT_SEGS, len(lab_segments))

        rng = np.random.default_rng(lab)
        sampled = [lab_segments[i] for i in rng.choice(len(lab_segments), n, replace=False)]

        n_cols = 5
        n_rows = (n + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(15, 2.2 * n_rows), sharey=True
        )
        axes = np.atleast_1d(axes).flatten()

        for idx, seg in enumerate(sampled):
            ax = axes[idx]
            sig = seg["feats"]
            t_rel = np.linspace(0, 1, len(sig))
            sig_norm = (sig - sig.min(axis=0)) / (
                sig.max(axis=0) - sig.min(axis=0) + 1e-8
            )

            for dim in range(6):
                ax.plot(
                    t_rel,
                    sig_norm[:, dim],
                    color=colors[dim],
                    alpha=0.8,
                    lw=0.8,
                    label=axes_names[dim] if idx == 0 else "",
                )
            ax.set_title(f"{seg['duration']}f", fontsize=7)
            ax.tick_params(axis="both", labelsize=5)
            ax.set_ylim(-0.1, 1.1)

        for idx in range(n, len(axes)):
            axes[idx].axis("off")

        if n > 0:
            axes[0].legend(loc="upper right", fontsize=5, ncol=2)

        fig.suptitle(
            f"state {target_state} sub-{lab} segments (K={k}, n={len(lab_segments)})",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(
            out_dir / f"state_{target_state:02d}_K{k}_sub{lab}_signals.png",
            dpi=150,
        )
        plt.close(fig)


def main():
    cfg = load_config()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    z_all, states_all, feats_all = load_all_data(cfg)

    summary = []

    for target_state in TARGET_STATES:
        print(f"\n=== state {target_state} ===")
        segments = extract_segments(z_all, states_all, feats_all, target_state)

        print(f"  有效 segment 数: {len(segments)}")

        if len(segments) == 0:
            continue

        X_scaled, best_labels, results = cluster_segments(segments, K_RANGE)

        if best_labels is None:
            continue

        best_k = max(results, key=lambda x: x[2])[0]

        plot_umap(X_scaled, best_labels, target_state, best_k, out_dir)
        plot_cluster_signals(segments, best_labels, target_state, best_k, out_dir)

        summary.append([target_state, best_k, max(r[2] for r in results)])

    np.savetxt(
        out_dir / "summary.csv",
        summary,
        delimiter=",",
        header="state,best_K,silhouette",
        comments="",
        fmt="%d,%d,%.4f",
    )

    print(f"\n分析完成，结果保存到 {out_dir}")
    for s, k, sil in summary:
        print(f"  state {s}: 最佳 sub-K={k}, silhouette={sil:.3f}")


if __name__ == "__main__":
    main()