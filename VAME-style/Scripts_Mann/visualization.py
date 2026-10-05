"""重构质量评估：用指定实验的模型对一段连续数据做重构对比。"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import PoseVQVAE
from training_dataset import load_raw_x
from utils import load_config, output_dir, latest_experiment_dir

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cfg = load_config()
    exp_dir = (output_dir(cfg) / sys.argv[1]) if len(sys.argv) > 1 \
        else latest_experiment_dir(cfg)

    ckpt = torch.load(exp_dir / "pose_vae.pt",
                      map_location=device, weights_only=False)
    model = PoseVQVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    npz_files = [p for p in sorted(output_dir(cfg).glob("*.npz"))
                 if not p.stem.endswith(("_latent", "_hmm"))]
    x_all = load_raw_x(npz_files[0])
    x = (x_all - ckpt["x_mean"]) / ckpt["x_std"]

    mid = len(x) // 2
    seg = torch.from_numpy(x[mid:mid + 2000].astype(np.float32))
    seg = seg.unsqueeze(0).to(device)

    with torch.no_grad():
        x_recon, mu, _ = model(seg)

    w = model.receptive_field - 1
    true = seg[0, w:].cpu().numpy()
    recon = x_recon[0, w:].cpu().numpy()
    z = mu[0, w:].cpu().numpy()

    names = ["sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "wz"]
    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    for i, ax in enumerate(axes[:5]):
        ax.plot(true[:, i], label="true", alpha=0.8)
        ax.plot(recon[:, i], label="recon", alpha=0.8)
        ax.set_ylabel(names[i])
        ax.legend(loc="upper right")
    for d in range(z.shape[1]):
        axes[5].plot(z[:, d], alpha=0.7)
    axes[5].set_ylabel("latent z")

    fig.tight_layout()
    fig.savefig(exp_dir / "eval_recon.png", dpi=150)
    plt.show()
    print(f"已保存到 {exp_dir / 'eval_recon.png'}")


if __name__ == "__main__":
    main()