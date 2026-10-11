"""多尺度/多分支 VAE 重建可视化。
支持任意分支配置和任意输入通道数。
用法：
    python recon_visualization.py [exp_dir] [recording_name]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import PoseVAE
from training_dataset import load_raw_x
from utils import load_config, output_dir, latest_experiment_dir

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_axis_names(in_features):
    """根据通道数返回轴名称，非 6 通道时用通用名。"""
    if in_features == 6:
        return ["ax", "ay", "az", "wx", "wy", "wz"]
    return [f"ch{i}" for i in range(in_features)]


def main():
    cfg = load_config()
    out = output_dir(cfg)

    # ---------- 参数解析 ----------
    arg1 = sys.argv[1] if len(sys.argv) > 1 else None

    if arg1 and (out / arg1).is_dir():
        exp_dir = out / arg1
        rec = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        exp_dir = latest_experiment_dir(cfg)
        rec = arg1

    # ---------- 加载模型 ----------
    ckpt = torch.load(
        exp_dir / "pose_vae.pt",
        map_location=device,
        weights_only=False,
    )
    model = PoseVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    branch_order = model.branch_order
    in_features = model.branch_configs[0]["channels"][0]  # 第一层输入通道 = 原始轴数
    axis_names = get_axis_names(in_features)

    # ---------- 选择 recording ----------
    npz_files = [
        p for p in sorted(out.glob("*.npz"))
        if not p.stem.endswith(("_latent", "_hmm", "_hsmm"))
    ]
    if not npz_files:
        raise FileNotFoundError("没有找到预处理 npz 文件")
    if rec is None:
        npz_path = npz_files[0]
    else:
        matches = [p for p in npz_files if p.stem == rec]
        if not matches:
            raise FileNotFoundError(f"没有找到 recording: {rec}")
        npz_path = matches[0]

    # ---------- 数据准备 ----------
    data = np.load(npz_path)
    x_raw = load_raw_x(npz_path)
    x = ((x_raw - ckpt["x_mean"]) / ckpt["x_std"]).astype(np.float32)

    mid = len(x) // 2
    seg_len = 2000
    seg = torch.from_numpy(x[mid : mid + seg_len]).unsqueeze(0).to(device)

    with torch.no_grad():
        recon_dict, mu, _ = model.forward_branches(seg)

    w = model.receptive_field - 1
    time = data["time"][mid + w : mid + seg_len]

    true = seg[0, w:].cpu().numpy()
    in_features = true.shape[1]          # ← 放这里
    axis_names = get_axis_names(in_features)
    pred_recon = recon_dict["recon"][0, w:].cpu().numpy()
    pred_branches = {
        name: recon_dict[name][0, w:].cpu().numpy()
        for name in branch_order
    }
    z = mu[0, w:].cpu().numpy()

    # ---------- 绘图 ----------
    n_axes = in_features
    fig, axes = plt.subplots(
        n_axes + 1, 1,
        figsize=(14, 2 + 2 * n_axes),
        sharex=True,
    )

    # 为每个分支分配颜色
    cmap = plt.cm.tab10
    branch_colors = {name: cmap(i % 10) for i, name in enumerate(branch_order)}

    for i, ax in enumerate(axes[:n_axes]):
        ax.plot(time, true[:, i], label="true", color="black", alpha=0.6, linewidth=1.5)
        ax.plot(time, pred_recon[:, i], label="recon", color="blue", alpha=0.7, linewidth=1.2)

        for name in branch_order:
            ax.plot(
                time,
                pred_branches[name][:, i],
                label=name,
                color=branch_colors[name],
                alpha=0.6,
                linewidth=1.0,
            )
        ax.set_ylabel(axis_names[i])
        ax.legend(loc="upper right", fontsize=8)

    # latent 面板：按分支分组着色
    start = 0
    for name in branch_order:
        zd = model.decoders[name].mlp[0].in_features  # 该分支的 latent 维度
        for d in range(start, start + zd):
            axes[n_axes].plot(
                time, z[:, d],
                alpha=0.6,
                color=branch_colors[name],
                label=name if d == start else "",
            )
        start += zd

    axes[n_axes].set_ylabel("latent z")
    axes[n_axes].set_xlabel("time (s)")
    axes[n_axes].legend(loc="upper right", fontsize=8, ncol=3)

    fig.suptitle(f"Reconstruction: {npz_path.stem}", y=1.0)
    fig.tight_layout()
    out_path = exp_dir / f"{npz_path.stem}_recon.png"
    fig.savefig(out_path, dpi=150)
    plt.show()
    print(f"已保存到 {out_path}")


if __name__ == "__main__":
    main()