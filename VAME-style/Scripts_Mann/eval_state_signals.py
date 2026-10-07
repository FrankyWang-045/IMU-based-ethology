"""评估 HMM/HSMM 每个 state 的时长分布与代表性原始信号图。

- 每个 state 随机抽取 40 个连续片段
- 支持 gaussian / categorical / hsmm 三种方法切换
- 排除包含 interp_mask=1 的片段
- 输出目录：F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest\eval
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_config, output_dir

METHOD = "hsmm"       # 可选：gaussian / categorical / hsmm
N_SAMPLES = 40            # 每个 state 抽取片段数
OUT_DIR = Path(r"F:\IMU_computational_ethology\Data\VAME-Style-Data\smalltest\eval")


def hmm_file_path(out, rec, method):
    """根据方法返回对应 HMM/HSMM 文件路径。"""
    if method == "categorical":
        return out / f"{rec}_hmm_cat.npz"
    elif method == "hsmm":
        return out / f"{rec}_hsmm.npz"
    return out / f"{rec}_hmm.npz"


def list_hmm_files(out, method):
    """列出当前方法下所有 HMM/HSMM 结果文件。"""
    if method == "categorical":
        return sorted(out.glob("*_hmm_cat.npz"))
    elif method == "hsmm":
        return sorted(out.glob("*_hsmm.npz"))
    return sorted(out.glob("*_hmm.npz"))


def load_raw_signals(csv_path):
    """读取原始 CSV 六轴信号。"""
    df = pd.read_csv(csv_path)
    cols = ["ax", "ay", "az", "wx", "wy", "wz"]
    return df["time"].values, df[cols].values, df["interp_mask"].values


def align_signals_to_hmm(time_raw, signals, interp, time_hmm):
    """将原始信号按时间对齐到 HMM/HSMM 状态轴。"""
    idx = np.searchsorted(time_raw, time_hmm)
    if not np.allclose(time_raw[idx], time_hmm):
        raise ValueError("原始信号时间与 HMM/HSMM 时间无法对齐")
    return signals[idx], interp[idx]


def find_segments(states):
    """返回所有连续状态片段 [(start, end, state), ...]。"""
    changes = np.where(np.diff(states) != 0)[0] + 1
    bounds = np.concatenate([[0], changes, [len(states)]])
    return [(bounds[i], bounds[i + 1], states[bounds[i]]) for i in range(len(bounds) - 1)]


def sample_segments(segments, n_samples, rng):
    """随机抽取最多 n_samples 个片段。"""
    if len(segments) <= n_samples:
        return segments
    idx = rng.choice(len(segments), size=n_samples, replace=False)
    return [segments[i] for i in idx]


def main():
    cfg = load_config()
    method = sys.argv[1] if len(sys.argv) > 1 else METHOD
    if method not in ("gaussian", "categorical", "hsmm"):
        raise ValueError("method 必须是 'gaussian'、'categorical' 或 'hsmm'")

    out = output_dir(cfg)
    raw_dir = Path(cfg["project"]["raw_dir"])
    n_states = cfg["hmm"]["n_states"]

    # 创建输出目录
    eval_dir = OUT_DIR / method
    eval_dir.mkdir(parents=True, exist_ok=True)
    signal_dir = eval_dir / "signals"
    signal_dir.mkdir(exist_ok=True)

    # 收集每个 state 的时长与片段
    durations_by_state = {s: [] for s in range(n_states)}
    segments_by_state = {s: [] for s in range(n_states)}
    segment_records = []

    for hmm_file in list_hmm_files(out, method):
        rec = hmm_file.stem.replace("_hmm_cat", "").replace("_hsmm", "").replace("_hmm", "")
        csv_path = raw_dir / f"{rec}.csv"
        if not csv_path.exists():
            print(f"跳过：找不到原始信号 {csv_path}")
            continue

        hmm = np.load(hmm_file)
        time_hmm, states, interp_hmm = hmm["time"], hmm["states"], hmm["interp_mask"]

        time_raw, signals, interp_raw = load_raw_signals(csv_path)
        signals_aligned, interp_aligned = align_signals_to_hmm(
            time_raw, signals, interp_raw, time_hmm
        )

        if len(signals_aligned) != len(states):
            print(f"警告：{rec} 信号与状态长度不一致，跳过")
            continue

        for s, e, state in find_segments(states):
            if interp_aligned[s:e].any():
                continue
            duration = time_hmm[e - 1] - time_hmm[s]
            durations_by_state[state].append(duration)
            segments_by_state[state].append(
                (rec, s, e, signals_aligned[s:e], time_hmm[s:e])
            )
            segment_records.append([rec, state, time_hmm[s], time_hmm[e - 1], duration])

    # ---------- 1. 时长分布统计 ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [durations_by_state[s] for s in range(n_states)]
    labels = [f"S{s}\n(n={len(d)})" for s, d in enumerate(data)]
    ax.boxplot(data, labels=labels)
    ax.set_xlabel("state")
    ax.set_ylabel("duration (s)")
    ax.set_title(f"State duration distribution ({method})")
    fig.tight_layout()
    fig.savefig(eval_dir / "duration_distribution.png", dpi=150)
    plt.close(fig)

    stats = []
    for s in range(n_states):
        d = np.array(durations_by_state[s])
        if len(d) == 0:
            stats.append([s, 0, 0, 0, 0, 0, 0])   # 7 列，修复 inhomogeneous 错误
        else:
            stats.append([s, len(d), d.mean(), np.median(d), d.std(), d.min(), d.max()])
    np.savetxt(
        eval_dir / "duration_stats.csv",
        stats,
        delimiter=",",
        header="state,count,mean,median,std,min,max",
        comments="",
        fmt="%.4f",
    )

    np.savetxt(
        eval_dir / "segment_records.csv",
        segment_records,
        delimiter=",",
        header="file,state,start_time,end_time,duration",
        comments="",
        fmt="%s",
    )

    # ---------- 2. 每个 state 的代表信号图 ----------
    rng = np.random.default_rng(0)
    axes_names = ["ax", "ay", "az", "wx", "wy", "wz"]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    for s in range(n_states):
        segs = segments_by_state[s]
        if len(segs) == 0:
            print(f"state {s:02d}: 无有效片段")
            continue

        sampled = sample_segments(segs, N_SAMPLES, rng)
        n_plot = len(sampled)
        n_cols = 8
        n_rows = (n_plot + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(18, 2.2 * n_rows), sharey=True
        )
        axes = np.atleast_1d(axes).flatten()

        for idx, (rec, s_idx, e_idx, sig, t) in enumerate(sampled):
            ax = axes[idx]
            t_rel = t - t[0]
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
            ax.set_title(
                f"{rec}\n{t[0]:.2f}s ({len(t)}f)",
                fontsize=7,
            )
            ax.tick_params(axis="both", labelsize=5)
            ax.set_ylim(-0.1, 1.1)

        for idx in range(n_plot, len(axes)):
            axes[idx].axis("off")

        if n_plot > 0:
            axes[0].legend(loc="upper right", fontsize=5, ncol=2)

        fig.suptitle(
            f"State {s} representative signals ({method}, n_total={len(segs)}, shown={n_plot}, per-axis normalized)",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(signal_dir / f"state_{s:02d}_signals.png", dpi=150)
        plt.close(fig)
        print(f"state {s:02d}: 共 {len(segs)} 个片段，绘制 {n_plot} 张")

    print(f"评估结果已保存到 {eval_dir}")


if __name__ == "__main__":
    main()