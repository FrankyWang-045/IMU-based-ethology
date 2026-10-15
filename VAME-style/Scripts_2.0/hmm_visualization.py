"""HSMM 状态序列与 latent 对照图 + 持续时间统计"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, output_dir, latest_experiment_dir


def duration_stats(states, fs):
    changes = np.where(np.diff(states) != 0)[0]
    durations = np.diff(np.concatenate([[0], changes, [len(states)]])) / fs
    return np.median(durations), (durations < 0.1).sum(), len(durations)


def main():
    cfg = load_config()
    out = output_dir(cfg)
    fs = cfg["preprocess"]["target_fs"]

    # 参数解析：第一个参数可能是 exp_dir，也可能是 rec
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None

    if arg1 and (out / arg1).is_dir():
        # 用法 1: python hmm_visualization.py exp_004 [rec] [t0] [window]
        exp_dir = out / arg1
        rec = sys.argv[2] if len(sys.argv) > 2 else None
        t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 610
        t1 = t0 + (float(sys.argv[4]) if len(sys.argv) > 4 else 10)
    else:
        # 用法 2: python hmm_visualization.py [rec] [t0] [window]
        exp_dir = latest_experiment_dir(cfg)
        rec = arg1
        t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 610
        t1 = t0 + (float(sys.argv[3]) if len(sys.argv) > 3 else 10)

    if rec is None:
        rec = sorted(exp_dir.glob("*_hsmm.npz"))[0].stem.replace("_hsmm", "")

    # 加载 latent 和 HSMM 结果
    latent = np.load(exp_dir / f"{rec}_latent.npz")
    z, time_z = latent["downstream"], latent["time"]

    hmm_h = np.load(exp_dir / f"{rec}_hsmm.npz")
    states_h, time_h = hmm_h["states"], hmm_h["time"]

    n_states = int(states_h.max()) + 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    m_z = (time_z >= t0) & (time_z < t1)
    for d in range(z.shape[1]):
        axes[0].plot(time_z[m_z], z[m_z, d], alpha=0.7, label=f"z{d}")
    axes[0].set_ylabel("latent z")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_title(f"latent + HSMM: {rec}")

    m_h = (time_h >= t0) & (time_h < t1)
    axes[1].scatter(
        time_h[m_h],
        states_h[m_h],
        c=states_h[m_h],
        cmap="tab20",
        s=4,
        vmin=-0.5,
        vmax=n_states - 0.5,
    )
    axes[1].set_ylabel("state")
    axes[1].set_yticks(range(n_states))
    axes[1].set_ylim(-0.5, n_states - 0.5)
    axes[1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(exp_dir / f"{rec}_hsmm_zoom.png", dpi=150)
    plt.show()

    med_h, short_h, total_h = duration_stats(states_h, fs)
    print(f"{rec} HSMM: 中位持续时间 {med_h:.2f}s, "
          f"<0.1s 碎片 {short_h}/{total_h}")


if __name__ == "__main__":
    main()