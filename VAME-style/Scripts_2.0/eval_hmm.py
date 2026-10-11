"""HSMM 性能评估与可视化：每个实验目录独立保存结果"""

import csv
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 让当前目录可被 import，用于加载 HSMM 类
sys.path.insert(0, str(Path(__file__).parent))
import hsmm_segment  # noqa: E402
from utils import load_config, output_dir, latest_experiment_dir  # noqa: E402


class MapMainUnpickler(pickle.Unpickler):
    """pickle 保存时 HSMM 在 __main__，加载时映射到 hsmm_segment 模块。"""

    def find_class(self, module, name):
        if module == "__main__":
            module = "hsmm_segment"
        return super().find_class(module, name)


def load_hsmm(exp_dir):
    with open(exp_dir / "hsmm_model.pkl", "rb") as f:
        return MapMainUnpickler(f).load()


def compute_segments(states):
    segments = []
    t = 0
    while t < len(states):
        s = t
        state = states[t]
        while t < len(states) and states[t] == state:
            t += 1
        segments.append((state, s, t - s))
    return segments


def duration_metrics(states, fs):
    segments = compute_segments(states)
    durations = np.array([d for _, _, d in segments], dtype=float) / fs
    return {
        "n_segments": len(segments),
        "median_duration_s": float(np.median(durations)),
        "mean_duration_s": float(np.mean(durations)),
        "p90_duration_s": float(np.percentile(durations, 90)),
        "fragments_lt_0.1s": int((durations < 0.1).sum()),
        "fragments_lt_0.2s": int((durations < 0.2).sum()),
        "fragments_lt_0.5s": int((durations < 0.5).sum()),
    }


def state_occupancy(states, n_states):
    occ = np.bincount(states, minlength=n_states) / len(states)
    entropy = -np.sum(occ * np.log(occ + 1e-10))
    return occ, float(entropy)


def segment_transition_matrix(segments, n_states):
    counts = np.ones((n_states, n_states))
    for idx in range(len(segments) - 1):
        i, _, _ = segments[idx]
        j, _, _ = segments[idx + 1]
        if i != j:
            counts[i, j] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def per_state_latent_stats(z, states, n_states):
    header = ["state", "n_frames"] + \
             [f"mean_z{d}" for d in range(z.shape[1])] + \
             [f"std_z{d}" for d in range(z.shape[1])]
    rows = []
    for s in range(n_states):
        mask = states == s
        if mask.sum() == 0:
            row = [s, 0] + [np.nan] * (2 * z.shape[1])
        else:
            zs = z[mask]
            row = [s, int(mask.sum())] + zs.mean(axis=0).tolist() + zs.std(axis=0).tolist()
        rows.append(row)
    return header, np.array(rows, dtype=object)


def plot_state_sequence(ax, time_h, states_h, time_z, z, t0, t1, n_states):
    m_z = (time_z >= t0) & (time_z < t1)
    n_plot = min(6, z.shape[1])
    for d in range(n_plot):
        ax[0].plot(time_z[m_z], z[m_z, d], alpha=0.7, label=f"z{d}")
    ax[0].set_ylabel("latent z")
    ax[0].legend(loc="upper right", ncol=3, fontsize=8)

    m_h = (time_h >= t0) & (time_h < t1)
    ax[1].scatter(
        time_h[m_h], states_h[m_h],
        c=states_h[m_h], cmap="tab20", s=4,
        vmin=-0.5, vmax=n_states - 0.5,
    )
    ax[1].set_ylabel("state")
    ax[1].set_yticks(range(n_states))
    ax[1].set_ylim(-0.5, n_states - 0.5)


def main():
    cfg = load_config()
    out = output_dir(cfg)
    exp_dir = (out / sys.argv[1]) if len(sys.argv) > 1 \
        else latest_experiment_dir(cfg)
    fs = cfg["preprocess"]["target_fs"]

    print(f"评估实验: {exp_dir}")

    hsmm = load_hsmm(exp_dir)
    n_states = hsmm.n_states

    latent_files = sorted(exp_dir.glob("*_latent.npz"))
    if not latent_files:
        raise FileNotFoundError(f"{exp_dir} 中没有 *_latent.npz")

    all_metrics = []
    all_segments = []

    for p in latent_files:
        rec = p.stem.replace("_latent", "")
        hsmm_file = exp_dir / f"{rec}_hsmm.npz"
        if not hsmm_file.exists():
            print(f"  跳过 {rec}，没有对应 hsmm 结果")
            continue

        data = np.load(p)
        z, time = data["downstream"], data["time"]
        states = np.load(hsmm_file)["states"]

        # 基础指标
        occ, entropy = state_occupancy(states, n_states)
        dm = duration_metrics(states, fs)
        dm["recording"] = rec
        dm["occupancy_entropy"] = entropy
        all_metrics.append(dm)

        segs = compute_segments(states)
        all_segments.extend(segs)

        # per-state latent 统计
        header, stats = per_state_latent_stats(z, states, n_states)
        np.savetxt(
            exp_dir / f"{rec}_state_latent_stats.csv",
            stats, delimiter=",", header=",".join(header),
            comments="", fmt="%s",
        )

        # 状态序列图（前 20 秒）
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        t_end = min(time[0] + 20, time[-1])
        plot_state_sequence(axes, time, states, time, z, time[0], t_end, n_states)
        axes[1].set_xlabel("time (s)")
        fig.suptitle(f"{rec} state sequence")
        fig.tight_layout()
        fig.savefig(exp_dir / f"{rec}_eval_stateseq.png", dpi=150)
        plt.close(fig)

    # 汇总指标 CSV
    if all_metrics:
        keys = list(all_metrics[0].keys())
        with open(exp_dir / "hmm_eval_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)

    # 汇总可视化
    durations_all = np.array([d / fs for _, _, d in all_segments])
    all_states = np.concatenate([
        np.load(exp_dir / f"{p.stem.replace('_latent', '')}_hsmm.npz")["states"]
        for p in latent_files
        if (exp_dir / f"{p.stem.replace('_latent', '')}_hsmm.npz").exists()
    ])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1) 时长分布
    axes[0, 0].hist(
        durations_all,
        bins=np.logspace(-2, 1, 50),
        color="steelblue", edgecolor="white",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].axvline(
        np.median(durations_all), color="red", linestyle="--",
        label=f"median={np.median(durations_all):.2f}s",
    )
    axes[0, 0].set_xlabel("duration (s)")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].legend()

    # 2) 片段间转移矩阵
    trans = segment_transition_matrix(all_segments, n_states)
    im = axes[0, 1].imshow(trans, cmap="viridis", aspect="auto")
    axes[0, 1].set_xlabel("to state")
    axes[0, 1].set_ylabel("from state")
    plt.colorbar(im, ax=axes[0, 1])

    # 3) 状态占比
    occ, _ = state_occupancy(all_states, n_states)
    axes[1, 0].bar(range(n_states), occ)
    axes[1, 0].set_xlabel("state")
    axes[1, 0].set_ylabel("occupancy")

    # 4) per-state latent 均值热图（用第一个文件）
    z0 = np.load(latent_files[0])["downstream"]
    states0 = np.load(
        exp_dir / f"{latent_files[0].stem.replace('_latent', '')}_hsmm.npz"
    )["states"]
    means = np.array([
        z0[states0 == s].mean(axis=0) if (states0 == s).any()
        else np.zeros(z0.shape[1])
        for s in range(n_states)
    ])
    im2 = axes[1, 1].imshow(means, cmap="coolwarm", aspect="auto")
    axes[1, 1].set_xlabel("latent dim")
    axes[1, 1].set_ylabel("state")
    plt.colorbar(im2, ax=axes[1, 1])

    fig.tight_layout()
    fig.savefig(exp_dir / "hmm_eval_summary.png", dpi=150)
    plt.close(fig)

    print(f"评估完成，结果保存到 {exp_dir}")


if __name__ == "__main__":
    main()