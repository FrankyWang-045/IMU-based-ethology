"""设备批次效应诊断脚本。

输出：
1. 每设备 latent 均值/方差；
2. LDA 判别轴及解释方差；
3. 每个 HSMM state 的设备偏置分数；
4. raw 6 轴每设备分布对比。
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import load_config, output_dir, latest_experiment_dir

AXES = ["ax", "ay", "az", "wx", "wy", "wz"]
MAX_SAMPLE = 50000          # LDA 和 raw 分布分析时每个设备最多采样帧数
RAW_SAMPLE_PER_DEVICE = 50000


def get_device_id(stem):
    """从文件名提取设备编号，例如 A1_C1_A1-6d_latent -> 6d。"""
    rec = stem.replace("_latent", "").replace("_hsmm", "")
    return rec.split("-")[-1]


def safe_jsd(p, q):
    """Jensen-Shannon divergence（对称、有界）。"""
    p = np.asarray(p, dtype=float) + 1e-10
    q = np.asarray(q, dtype=float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def load_latent_and_states(exp_dir):
    latent_files = sorted(exp_dir.glob("*_latent.npz"))
    records = []
    for p in latent_files:
        data = np.load(p)
        z = data["downstream"]
        z = z[np.isfinite(z).all(axis=1)]

        hsmm_file = exp_dir / f"{p.stem.replace('_latent', '')}_hsmm.npz"
        states = None
        if hsmm_file.exists():
            states = np.load(hsmm_file)["states"]

        records.append({
            "name": p.stem.replace("_latent", ""),
            "device": get_device_id(p.stem),
            "z": z,
            "states": states,
        })
    return records


def latent_device_stats(records, out_dir):
    """每设备 latent 均值/标准差。"""
    rows = []
    for r in records:
        z = r["z"]
        for d in range(z.shape[1]):
            rows.append({
                "recording": r["name"],
                "device": r["device"],
                "dim": d,
                "mean": float(z[:, d].mean()),
                "std": float(z[:, d].std()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "latent_per_device_stats.csv", index=False)

    # 汇总：每设备所有 recording 的聚合均值和标准差
    agg_rows = []
    devices = sorted(set(r["device"] for r in records))
    for dev in devices:
        zs = [r["z"] for r in records if r["device"] == dev]
        z_cat = np.concatenate(zs, axis=0)
        for d in range(z_cat.shape[1]):
            agg_rows.append({
                "device": dev,
                "dim": d,
                "mean": float(z_cat[:, d].mean()),
                "std": float(z_cat[:, d].std()),
                "n_frames": len(z_cat),
            })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / "latent_device_aggregated.csv", index=False)

    # 热力图：device x dim 的均值
    pivot = agg_df.pivot(index="device", columns="dim", values="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"z{d}" for d in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean latent per device")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "latent_device_mean_heatmap.png", dpi=150)
    plt.close(fig)

    return agg_df


def lda_analysis(records, out_dir):
    """LDA：找最能区分设备的方向。"""
    devices = sorted(set(r["device"] for r in records))
    if len(devices) < 2:
        print("设备数不足 2，跳过 LDA")
        return None

    # 分层采样，避免某个 recording/设备过大
    X_list, y_list = [], []
    for dev in devices:
        dev_zs = [r["z"] for r in records if r["device"] == dev]
        z_cat = np.concatenate(dev_zs, axis=0)
        n = min(len(z_cat), MAX_SAMPLE // len(devices))
        idx = np.random.choice(len(z_cat), n, replace=False)
        X_list.append(z_cat[idx])
        y_list.append(np.full(n, dev))
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, y)

    # 解释方差
    ev_ratio = lda.explained_variance_ratio_
    n_comp = min(len(ev_ratio), 5)
    print(f"LDA 解释方差比: {ev_ratio[:n_comp]}")

    # 保存判别轴 loadings
    scalings = lda.scalings_  # (n_dims, n_components)
    loadings = pd.DataFrame(
        scalings[:, :n_comp],
        index=[f"z{d}" for d in range(X.shape[1])],
        columns=[f"LD{i+1}" for i in range(n_comp)],
    )
    loadings.to_csv(out_dir / "lda_loadings.csv")

    # 图 1：解释方差
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n_comp + 1), ev_ratio[:n_comp])
    ax.set_xlabel("LD")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("LDA explained variance by device")
    fig.tight_layout()
    fig.savefig(out_dir / "lda_explained_variance.png", dpi=150)
    plt.close(fig)

    # 图 2：前两个 LD 投影
    if X_lda.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for dev in devices:
            mask = y == dev
            ax.scatter(
                X_lda[mask, 0],
                X_lda[mask, 1],
                label=dev,
                alpha=0.3,
                s=4,
            )
        ax.set_xlabel("LD1")
        ax.set_ylabel("LD2")
        ax.set_title("LDA projection colored by device")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "lda_projection.png", dpi=150)
        plt.close(fig)

    return lda, ev_ratio, loadings


def state_device_bias(records, n_states, out_dir):
    """每个 state 在不同设备中的分布偏置。"""
    if any(r["states"] is None for r in records):
        print("部分 recording 没有 HSMM 结果，跳过 state 偏置分析")
        return None

    devices = sorted(set(r["device"] for r in records))
    state_device_counts = np.zeros((n_states, len(devices)), dtype=float)

    for r in records:
        dev_idx = devices.index(r["device"])
        states = r["states"]
        occ = np.bincount(states, minlength=n_states)
        state_device_counts[:, dev_idx] += occ

    # 每 state 在不同设备中的分布
    state_device_dist = state_device_counts / state_device_counts.sum(axis=1, keepdims=True)

    # 全局设备分布
    global_device_dist = state_device_counts.sum(axis=0)
    global_device_dist = global_device_dist / global_device_dist.sum()

    # 每个 state 的偏置分数
    rows = []
    for s in range(n_states):
        p = state_device_dist[s]
        q = global_device_dist
        jsd = safe_jsd(p, q)
        chi2 = np.sum((p - q) ** 2 / (q + 1e-10))
        dominant = devices[int(np.argmax(p))]
        rows.append({
            "state": s,
            "jsd": jsd,
            "chi2": chi2,
            "dominant_device": dominant,
            "max_device_ratio": float(p.max()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "state_device_bias.csv", index=False)

    # 图：state 偏置分数
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["state"], df["jsd"], color="steelblue")
    ax.set_xlabel("state")
    ax.set_ylabel("JSD vs global device dist")
    ax.set_title("State device bias (higher = more device-specific)")
    fig.tight_layout()
    fig.savefig(out_dir / "state_device_bias.png", dpi=150)
    plt.close(fig)

    # 热力图：state x device 分布
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(state_device_dist, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(devices)))
    ax.set_xticklabels(devices)
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels(range(n_states))
    ax.set_xlabel("device")
    ax.set_ylabel("state")
    ax.set_title("P(device | state)")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "state_device_heatmap.png", dpi=150)
    plt.close(fig)

    print("Top 5 most device-biased states:")
    print(df.sort_values("jsd", ascending=False).head().to_string(index=False))

    return df


def raw_device_distribution(cfg, records, out_dir):
    """raw 6 轴每设备分布对比。"""
    out = output_dir(cfg)
    device_samples = {dev: [] for dev in sorted(set(r["device"] for r in records))}

    for r in records:
        raw_file = out / f"{r['name']}.npz"
        if not raw_file.exists():
            continue
        data = np.load(raw_file)
        feats = data["features"]
        if len(feats) == 0:
            continue
        # 随机采样
        n = min(len(feats), RAW_SAMPLE_PER_DEVICE // len(device_samples))
        idx = np.random.choice(len(feats), n, replace=False)
        device_samples[r["device"]].append(feats[idx])

    if not any(device_samples.values()):
        print("未找到 raw npz 文件，跳过 raw 分布分析")
        return None

    # 汇总统计
    rows = []
    for dev, samples in device_samples.items():
        if not samples:
            continue
        x = np.concatenate(samples, axis=0)
        for i, name in enumerate(AXES):
            rows.append({
                "device": dev,
                "axis": name,
                "mean": float(x[:, i].mean()),
                "std": float(x[:, i].std()),
                "median": float(np.median(x[:, i])),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "raw_device_stats.csv", index=False)

    # 绘制每个轴的分布
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes = axes.flatten()
    for i, name in enumerate(AXES):
        ax = axes[i]
        for dev, samples in device_samples.items():
            if not samples:
                continue
            x = np.concatenate(samples, axis=0)
            ax.hist(
                x[:, i],
                bins=100,
                alpha=0.5,
                label=dev,
                density=True,
                histtype="step",
            )
        ax.set_title(name)
        ax.legend(fontsize=7)
    fig.suptitle("Raw axis distribution by device")
    fig.tight_layout()
    fig.savefig(out_dir / "raw_device_distributions.png", dpi=150)
    plt.close(fig)

    return df


def main():
    cfg = load_config()
    out = output_dir(cfg)

    parser = argparse.ArgumentParser(description="设备批次效应诊断")
    parser.add_argument("--exp", type=str, default=None,
                        help="实验目录，如 exp_004 或 004；默认最新实验")
    args = parser.parse_args()

    if args.exp:
        exp_name = args.exp if str(args.exp).startswith("exp_") else f"exp_{int(args.exp):03d}"
        exp_dir = out / exp_name
        if not exp_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {exp_dir}")
    else:
        exp_dir = latest_experiment_dir(cfg)

    print(f"使用实验目录: {exp_dir}")

    out_dir = exp_dir / "device_bias"
    out_dir.mkdir(exist_ok=True)

    records = load_latent_and_states(exp_dir)
    print(f"加载 {len(records)} 个 recording")

    n_states = cfg["hmm"]["n_states"]

    # 1. latent 每设备统计
    print("\n[1/4] latent 每设备统计...")
    latent_device_stats(records, out_dir)

    # 2. LDA
    print("\n[2/4] LDA 设备判别分析...")
    lda_analysis(records, out_dir)

    # 3. state 设备偏置
    print("\n[3/4] state 设备偏置...")
    state_device_bias(records, n_states, out_dir)

    # 4. raw 分布
    print("\n[4/4] raw 6 轴每设备分布...")
    raw_device_distribution(cfg, records, out_dir)

    print(f"\n诊断结果保存到: {out_dir}")


if __name__ == "__main__":
    main()