"""HMM 状态序列与 latent 对照图 + 持续时间统计。"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from utils import load_config, latent_path, hmm_path


def main():
    cfg = load_config()
    rec = sys.argv[1] if len(sys.argv) > 1 else None

    hmm_files = sorted(hmm_path(cfg, p.stem.replace("_latent", ""))
                       for p in [])
    # 找第一个可用的 hmm 文件
    out = latent_path(cfg, "*").parent
    candidates = sorted(out.glob("*_hmm.npz"))
    if rec is None:
        rec = candidates[0].stem.replace("_hmm", "")

    hmm = np.load(hmm_path(cfg, rec))
    lat = np.load(latent_path(cfg, rec))
    states, time, z = hmm["states"], hmm["time"], lat["downstream"]

    # --- 图 A：20 秒窗口对照 ---
    t0, t1 = 600, 630

    mask = (time >= t0) & (time < t1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(time[mask], z[mask])
    axes[0].set_ylabel("latent z")
    axes[1].scatter(time[mask], states[mask], c=states[mask],
                    cmap="tab10", s=2, vmin=-0.5, vmax=9.5)
    axes[1].set_ylabel("state")
    axes[1].set_yticks(range(cfg["hmm"]["n_states"]))
    fig.tight_layout()
    fig.savefig(out / f"{rec}_hmm_zoom.png", dpi=150)
    plt.show()

    # --- 持续时间统计 ---
    changes = np.where(np.diff(states) != 0)[0]
    durations = np.diff(np.concatenate([[0], changes, [len(states)]])) / 100
    print(f"{rec}: 中位持续时间 {np.median(durations):.2f}s, "
          f"<0.1s 碎片 {(durations < 0.1).sum()}/{len(durations)}")


if __name__ == "__main__":
    main()