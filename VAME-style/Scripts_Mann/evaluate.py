"""VQ-VAE 定量评估：逐文件逐轴 R² + 重建对比图 + 残差分析。"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import PoseVQVAE
from training_dataset import load_raw_x
from utils import load_config, output_dir, latest_experiment_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
AXES = ["sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "wz"]
SEG_LEN = 2000          # 重建图窗口长度（帧）


def r2_score(true, pred):
    """逐列 R²，返回 shape (n_axes,)。"""
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / ss_tot


def main():
    cfg = load_config()
    exp_dir = (output_dir(cfg) / sys.argv[1]) if len(sys.argv) > 1 \
        else latest_experiment_dir(cfg)
    print(f"评估模型: {exp_dir}")

    ckpt = torch.load(exp_dir / "pose_vae.pt",
                      map_location=device, weights_only=False)
    model = PoseVQVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    w = model.receptive_field - 1

    npz_files = [p for p in sorted(output_dir(cfg).glob("*.npz"))
                 if not p.stem.endswith(("_latent", "_hmm"))]

    # ---------- 逐文件 R² ----------
    rows = []
    recons = {}          # 缓存第一个文件的重构，供画图
    for p in npz_files:
        x = (load_raw_x(p) - ckpt["x_mean"]) / ckpt["x_std"]
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            x_recon, _, _, _ = model(x_t)
        true = x[w:]
        pred = x_recon[0, w:].cpu().numpy()
        r2 = r2_score(true, pred)
        rows.append([p.stem, *r2])
        print(f"{p.stem}: " + "  ".join(f"{a}={v:.3f}" for a, v in zip(AXES, r2)))
        if not recons:
            recons = {"name": p.stem, "true": true, "pred": pred}

    header = "file," + ",".join(AXES)
    np.savetxt(exp_dir / "r2_per_axis.csv", np.array(rows)[:, 1:].astype(float),
               delimiter=",", header=header + " (行顺序: " +
               ";".join(r[0] for r in rows) + ")", comments="", fmt="%.4f")

    # ---------- 重建对比图（第一个文件，中段窗口） ----------
    true, pred = recons["true"], recons["pred"]
    mid = len(true) // 2
    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(true[mid:mid + SEG_LEN, i], label="true", alpha=0.8)
        ax.plot(pred[mid:mid + SEG_LEN, i], label="recon", alpha=0.8)
        ax.set_ylabel(AXES[i]); ax.legend(loc="upper right")
    fig.suptitle(f"Reconstruction: {recons['name']}")
    fig.tight_layout()
    fig.savefig(exp_dir / "eval_recon.png", dpi=150)
    plt.close(fig)

    # ---------- 残差分析（第一个文件，全序列直方图 + 窗口时序） ----------
    resid = true - pred                      # (T, 5)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].plot(np.abs(resid[mid:mid + SEG_LEN]).mean(axis=1))
    axes[0].set_ylabel("|residual| (5轴均值)")
    axes[0].set_title(f"Residual: {recons['name']}")
    axes[1].hist(np.abs(resid).ravel(), bins=100, log=True)
    axes[1].set_xlabel("|residual|"); axes[1].set_ylabel("count (log)")
    fig.tight_layout()
    fig.savefig(exp_dir / "eval_residual.png", dpi=150)
    plt.close(fig)

    print(f"评估产物已保存到 {exp_dir}")


if __name__ == "__main__":
    main()