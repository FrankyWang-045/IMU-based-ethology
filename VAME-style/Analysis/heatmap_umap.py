# -*- coding: utf-8 -*-
"""
根据 config.yaml 中的 output_dir，读取最新 exp 的 HSMM 结果，
从原始 *.npz 中按时间戳对齐并提取状态段特征，
构造 4 通道（acc_mag-1, gyro_x, gyro_y, gyro_z），
提取 mean/var/zcr/autocorr + duration 后跨文件聚合，
绘制状态热图，并对所有片段做 GMM 聚类 + UMAP 可视化。
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# 尝试导入 umap，若未安装给出友好提示
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ==================== 可配置参数 ====================

CONFIG_PATH = Path(r"F:\IMU_computational_ethology\IMU-based-ethology\VAME-style\config.yaml")

# 特征提取参数
MIN_SEGMENT_LENGTH = 10
MAX_LAG_FRAMES = 50
TARGET_FS = 25.0                   # 从 config 读取更严谨，这里默认 25Hz

# GMM + UMAP 参数
GMM_N_COMPONENTS = 43               # GMM 聚类数，可命令行覆盖
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
UMAP_RANDOM_STATE = 42

# 图形参数
FIGSIZE_HEATMAP = (14, 10)
FIGSIZE_UMAP = (16, 7)
DPI = 150

# ===================================================


def parse_config(config_path: Path) -> dict:
    """简单解析 YAML，获取 project.output_dir 和 target_fs。"""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception as e:
        print(f"PyYAML 读取失败 ({e})，使用正则回退解析...")
        with open(config_path, "r", encoding="utf-8") as f:
            text = f.read()
        output_dir = re.search(r"output_dir:\s*(.+)", text)
        target_fs = re.search(r"target_fs:\s*([\d.]+)", text)
        return {
            "project": {"output_dir": output_dir.group(1).split("#")[0].strip() if output_dir else None},
            "preprocess": {"target_fs": float(target_fs.group(1)) if target_fs else TARGET_FS},
        }


def find_latest_exp(output_dir: Path) -> Path:
    """在 output_dir 下找到数字最大的 exp_xxx 文件夹。"""
    exp_dirs = [d for d in output_dir.glob("exp_*") if d.is_dir()]
    if not exp_dirs:
        raise FileNotFoundError(f"在 {output_dir} 下未找到 exp_* 文件夹")
    
    def get_num(p: Path):
        m = re.search(r"(\d+)$", p.name)
        return int(m.group(1)) if m else -1
    
    return max(exp_dirs, key=get_num)


def find_file_pairs(output_dir: Path, exp_dir: Path):
    pairs = []
    for hsmm_path in sorted(exp_dir.glob("*_hsmm.npz")):
        key = hsmm_path.stem.replace("_hsmm", "")
        raw_path = output_dir / f"{key}.npz"
        if not raw_path.exists():
            print(f"[WARNING] 跳过 {hsmm_path.name}: 未找到原始数据 {raw_path}")
            continue
        pairs.append({"key": key, "hsmm_path": hsmm_path, "raw_path": raw_path})
    return pairs


def align_raw_to_hsmm(raw_time: np.ndarray, hsmm_time: np.ndarray):
    raw_indices = np.searchsorted(raw_time, hsmm_time)
    raw_indices = np.clip(raw_indices, 0, len(raw_time) - 1)
    left = np.maximum(raw_indices - 1, 0)
    right = raw_indices
    closer = np.abs(raw_time[left] - hsmm_time) <= np.abs(raw_time[right] - hsmm_time)
    return np.where(closer, left, right)


def build_channels(raw_features: np.ndarray):
    acc = raw_features[:, :3]
    gyro = raw_features[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    return np.concatenate([acc_mag - 1.0, gyro], axis=1)


def extract_segment_features(segment: np.ndarray, duration_sec: float):
    """
    segment: (L, 4)
    返回: (17,) 特征向量 [16 个通道特征 + duration_sec]
    """
    L, C = segment.shape
    feats = np.zeros((C, 4))
    
    feats[:, 0] = np.mean(segment, axis=0)
    feats[:, 1] = np.var(segment, axis=0)
    feats[:, 2] = np.mean(np.diff(np.sign(segment), axis=0) != 0, axis=0)
    
    for c in range(C):
        x = segment[:, c] - np.mean(segment[:, c])
        if np.var(x) < 1e-12 or L < 3:
            feats[c, 3] = 0.0
            continue
        max_lag = min(MAX_LAG_FRAMES, L // 2)
        corrs = []
        for lag in range(1, max_lag + 1):
            corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            if not np.isnan(corr):
                corrs.append(corr)
        feats[c, 3] = max(corrs) if corrs else 0.0
    
    return np.concatenate([feats.ravel(), [duration_sec]])


def get_segments(states: np.ndarray, features4d: np.ndarray, hsmm_time: np.ndarray,
                 min_len: int = MIN_SEGMENT_LENGTH, target_fs: float = TARGET_FS):
    """
    提取每个连续段的 17 维特征 + state 标签。
    返回 list of (feature_vector, state_id)
    """
    results = []
    if len(states) == 0:
        return results
    
    boundaries = np.where(np.diff(states) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(states)]])
    
    for s, e in zip(starts, ends):
        length = e - s
        if length < min_len:
            continue
        state = int(states[s])
        seg = features4d[s:e]
        duration_sec = length / target_fs
        feats = extract_segment_features(seg, duration_sec)
        results.append({"features": feats, "state": state, "length": length, "duration_sec": duration_sec})
    
    return results


def aggregate_state_features(segments_by_state: dict):
    state_ids = sorted(segments_by_state.keys())
    state_matrix = np.array([np.median(np.stack(segments_by_state[s]), axis=0)
                             for s in state_ids])
    return np.array(state_ids), state_matrix


def plot_heatmap(state_ids: np.ndarray, state_matrix: np.ndarray, save_path: Path = None):
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(state_matrix)
    
    channel_names = ["acc_mag-1", "gyro_x", "gyro_y", "gyro_z"]
    feature_names = ["mean", "var", "zcr", "autocorr"]
    x_labels = [f"{c}_{f}" for c in channel_names for f in feature_names]
    # 最后一列是 duration
    x_labels.append("duration_sec")
    
    plt.figure(figsize=FIGSIZE_HEATMAP)
    sns.heatmap(
        normed,
        xticklabels=x_labels,
        yticklabels=state_ids,
        cmap="viridis",
        cbar_kws={"label": "normalized value"},
        linewidths=0.1,
    )
    plt.title(f"State Feature Heatmap (n_states={len(state_ids)})")
    plt.xlabel("Features")
    plt.ylabel("HSMM State")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"[INFO] 热图已保存: {save_path}")
    plt.show()


def plot_umap(segment_matrix: np.ndarray, gmm_labels: np.ndarray, state_labels: np.ndarray,
              save_path: Path = None):
    """
    绘制两张 UMAP 子图：左图按 GMM 聚类上色，右图按 HSMM state 上色。
    """
    if not UMAP_AVAILABLE:
        print("[ERROR] 未安装 umap-learn，请先运行: pip install umap-learn")
        return
    
    # 标准化后 UMAP
    X = StandardScaler().fit_transform(segment_matrix)
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
        n_components=2,
    )
    emb = reducer.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_UMAP)
    
    # 左图：GMM 聚类
    scatter1 = axes[0].scatter(
        emb[:, 0], emb[:, 1],
        c=gmm_labels,
        cmap="tab20",
        s=8,
        alpha=0.6,
        edgecolors="none",
    )
    axes[0].set_title(f"UMAP colored by GMM cluster (k={len(np.unique(gmm_labels))})")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    plt.colorbar(scatter1, ax=axes[0], label="GMM cluster")
    
    # 右图：HSMM state
    scatter2 = axes[1].scatter(
        emb[:, 0], emb[:, 1],
        c=state_labels,
        cmap="tab20",
        s=8,
        alpha=0.6,
        edgecolors="none",
    )
    axes[1].set_title("UMAP colored by HSMM state")
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    plt.colorbar(scatter2, ax=axes[1], label="HSMM state")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"[INFO] UMAP 图已保存: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--gmm_k", type=int, default=GMM_N_COMPONENTS,
                        help="GMM 聚类数")
    parser.add_argument("--save_heatmap", type=str, default=None)
    parser.add_argument("--save_umap", type=str, default=None)
    args = parser.parse_args()
    
    cfg = parse_config(Path(args.config))
    output_dir = Path(cfg["project"]["output_dir"])
    target_fs = cfg.get("preprocess", {}).get("target_fs", TARGET_FS)
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] target_fs: {target_fs}")
    
    exp_dir = find_latest_exp(output_dir)
    print(f"[INFO] 使用最新 exp: {exp_dir.name}")
    
    pairs = find_file_pairs(output_dir, exp_dir)
    if not pairs:
        raise RuntimeError("未找到有效的 HSMM/原始数据文件对")
    print(f"[INFO] 找到 {len(pairs)} 个文件对")
    
    all_segments = []  # list of dict
    for p in pairs:
        print(f"  处理: {p['key']}")
        hsmm = np.load(p["hsmm_path"])
        raw = np.load(p["raw_path"])
        
        raw_indices = align_raw_to_hsmm(raw["time"], hsmm["time"])
        features4d = build_channels(raw["features"][raw_indices])
        
        segs = get_segments(hsmm["states"], features4d, hsmm["time"],
                            target_fs=target_fs)
        all_segments.extend(segs)
    
    print(f"[INFO] 总共提取 {len(all_segments)} 个有效片段")
    
    # ===== 状态热图 =====
    segments_by_state = {}
    for seg in all_segments:
        segments_by_state.setdefault(seg["state"], []).append(seg["features"])
    
    state_ids, state_matrix = aggregate_state_features(segments_by_state)
    print(f"[INFO] 有效状态数: {len(state_ids)}")
    
    save_heatmap = Path(args.save_heatmap) if args.save_heatmap else None
    plot_heatmap(state_ids, state_matrix, save_path=save_heatmap)
    
    # ===== GMM 聚类 + UMAP =====
    segment_matrix = np.stack([seg["features"] for seg in all_segments])
    state_labels = np.array([seg["state"] for seg in all_segments])
    
    print(f"[INFO] GMM 聚类中 (k={args.gmm_k})...")
    gmm = GaussianMixture(n_components=args.gmm_k, random_state=42, n_init=10)
    gmm_labels = gmm.fit_predict(StandardScaler().fit_transform(segment_matrix))
    
    print(f"[INFO] UMAP 降维并绘制...")
    save_umap = Path(args.save_umap) if args.save_umap else None
    plot_umap(segment_matrix, gmm_labels, state_labels, save_path=save_umap)


if __name__ == "__main__":
    main()