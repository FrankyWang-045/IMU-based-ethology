"""HMM/HSMM 状态序列与 latent 对照图 + 持续时间统计。

同时绘制 Gaussian HMM、Categorical HMM 和 HSMM 的状态序列。

用法: python hmm_visualization.py [文件名] [起始秒] [窗口时长秒]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, output_dir, latent_path, hmm_path, hmm_cat_path, hmm_hsmm_path


def plot_hmm_states(ax, time, states, t0, t1, n_states, title):
    m = (time >= t0) & (time < t1)
    ax.scatter(
        time[m],
        states[m],
        c=states[m],
        cmap="tab20",
        s=4,
        vmin=-0.5,
        vmax=n_states - 0.5,
    )
    ax.set_ylabel("state")
    ax.set_yticks(range(n_states))
    ax.set_title(title)
    ax.set_ylim(-0.5, n_states - 0.5)


def duration_stats(states):
    changes = np.where(np.diff(states) != 0)[0]
    durations = np.diff(np.concatenate([[0], changes, [len(states)]])) / 100
    return np.median(durations), (durations < 0.1).sum(), len(durations)


def main():
    cfg = load_config()
    out = output_dir(cfg)

    # 参数：文件、时间窗口
    rec = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    t1 = t0 + (float(sys.argv[3]) if len(sys.argv) > 3 else 20)

    if rec is None:
        rec = sorted(out.glob("*_hmm.npz"))[0].stem.replace("_hmm", "")

    # ---------- 加载 latent ----------
    latent = np.load(latent_path(cfg, rec))
    z, time_z = latent["downstream"], latent["time"]

    # ---------- 加载三种模型结果 ----------
    hmm_g = np.load(hmm_path(cfg, rec))
    states_g, time_g = hmm_g["states"], hmm_g["time"]

    hmm_c = np.load(hmm_cat_path(cfg, rec))
    states_c, time_c = hmm_c["states"], hmm_c["time"]

    hmm_h = np.load(hmm_hsmm_path(cfg, rec))
    states_h, time_h = hmm_h["states"], hmm_h["time"]

    # ---------- 图：latent + 三种 HMM ----------
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    m_z = (time_z >= t0) & (time_z < t1)
    for d in range(z.shape[1]):
        axes[0].plot(time_z[m_z], z[m_z, d], alpha=0.7, label=f"z{d}")
    axes[0].set_ylabel("latent z")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_title(f"latent + HMM/HSMM states: {rec}")

    n_states = cfg["hmm"]["n_states"]
    plot_hmm_states(axes[1], time_g, states_g, t0, t1, n_states, "Gaussian HMM")
    plot_hmm_states(axes[2], time_c, states_c, t0, t1, n_states, "Categorical HMM")
    plot_hmm_states(axes[3], time_h, states_h, t0, t1, n_states, "HSMM")

    axes[3].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / f"{rec}_hmm_compare.png", dpi=150)
    plt.show()

    # ---------- 持续时间统计 ----------
    med_g, short_g, total_g = duration_stats(states_g)
    med_c, short_c, total_c = duration_stats(states_c)
    med_h, short_h, total_h = duration_stats(states_h)

    print(f"{rec} Gaussian HMM : 中位持续时间 {med_g:.2f}s, <0.1s 碎片 {short_g}/{total_g}")
    print(f"{rec} Categorical HMM: 中位持续时间 {med_c:.2f}s, <0.1s 碎片 {short_c}/{total_c}")
    print(f"{rec} HSMM           : 中位持续时间 {med_h:.2f}s, <0.1s 碎片 {short_h}/{total_h}")


if __name__ == "__main__":
    main()