from pathlib import Path
import argparse

# ---------- 修复 umap-learn 与新版 scikit-learn 兼容性 ----------
import sklearn.utils
from sklearn.utils import check_array as _original_check_array


def _check_array_patched(X, *args, **kwargs):
    force = kwargs.pop("force_all_finite", None)
    if "ensure_all_finite" not in kwargs:
        if force is True:
            kwargs["ensure_all_finite"] = True
        elif force is False:
            kwargs["ensure_all_finite"] = "allow-nan"
        elif force is not None:
            kwargs["ensure_all_finite"] = force
    return _original_check_array(X, *args, **kwargs)


sklearn.utils.check_array = _check_array_patched
# --------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import umap

from utils import load_config, output_dir


def plot_states_umap(
    cfg,
    recording_name,
    t_start,
    t_end,
    latent_subdir="latent",
    cmap="tab20",
    figsize=(16, 9),
    dpi=100,
):
    out = output_dir(cfg)
    latent_dir = out / latent_subdir

    latent_path = latent_dir / f"{recording_name}_latent.npz"
    states_path = latent_dir / f"{recording_name}_states.npz"

    if not latent_path.exists():
        raise FileNotFoundError(f"未找到 latent: {latent_path}")
    if not states_path.exists():
        raise FileNotFoundError(f"未找到状态标签，请先运行 fit_hdpslds.py: {states_path}")

    latent_data = np.load(latent_path)
    state_data = np.load(states_path)

    latent = latent_data["latent"]
    states = state_data["states"]
    time = latent_data["time"]

    # 时间截取
    mask = (time >= t_start) & (time <= t_end)
    if not mask.any():
        raise ValueError(f"时间范围 [{t_start}, {t_end}] 内无数据")

    segment_latent = latent[mask]
    segment_states = states[mask]

    print(f"UMAP 降维: {segment_latent.shape[0]} 样本 x {segment_latent.shape[1]} 维")
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(segment_latent)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=segment_states,
        cmap=cmap,
        s=20,
        alpha=0.8,
        edgecolors="none",
    )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{recording_name} | HDP-SLDS states on UMAP | t={t_start:.1f}s-{t_end:.1f}s")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("state")

    save_path = latent_dir / f"{recording_name}_umap_states_{t_start:.1f}_{t_end:.1f}.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"已保存: {save_path}")

    unique, counts = np.unique(segment_states, return_counts=True)
    print(f"\n状态分布:")
    for s, c in zip(unique, counts):
        print(f"  state {s}: {c} 帧 ({100*c/len(segment_states):.1f}%)")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="指定时间段 UMAP 状态上色可视化")
    parser.add_argument("--recording", required=True, help="recording 名")
    parser.add_argument("--start", type=float, required=True, help="起始时间戳（秒）")
    parser.add_argument("--end", type=float, required=True, help="结束时间戳（秒）")
    args = parser.parse_args()

    cfg = load_config()
    plot_states_umap(cfg, args.recording, args.start, args.end)


if __name__ == "__main__":
    main()