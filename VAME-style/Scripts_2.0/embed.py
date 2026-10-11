"""批量 latent 提取：结果保存到对应实验目录"""

import sys

import numpy as np
import torch

from model import PoseVAE
from utils import ROOT, load_config, output_dir, latest_experiment_dir

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(exp_dir, device):
    ckpt = torch.load(exp_dir / "pose_vae.pt",
                      map_location=device, weights_only=False)
    model = PoseVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def build_vae_input(feats, x_mean, x_std):
    return ((feats - x_mean) / x_std).astype(np.float32)


def extract_latent(model, x, device):
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        _, mu, _ = model(x_t)
    return mu[0].cpu().numpy()


def embed_file(npz_path, model, ckpt, out_path):
    data = np.load(npz_path)
    feats = data["features"]
    x = build_vae_input(feats, ckpt["x_mean"], ckpt["x_std"])
    z = extract_latent(model, x, device)

    warmup = model.receptive_field - 1
    z = z[warmup:].astype(np.float32)
    time = data["time"][warmup:]
    interp_mask = data["interp_mask"][warmup:]

    np.savez_compressed(out_path, downstream=z,
                        time=time, interp_mask=interp_mask)
    print(f"  {npz_path.stem}: {z.shape}")


def main():
    cfg = load_config()
    out = output_dir(cfg)
    exp_dir = (out / sys.argv[1]) if len(sys.argv) > 1 \
        else latest_experiment_dir(cfg)
    print(f"使用模型: {exp_dir}")

    model, ckpt = load_model(exp_dir, device)

    # 预处理 npz 仍在 output_dir 根目录
    npz_files = [p for p in sorted(out.glob("*.npz"))
                 if not p.stem.endswith(("_latent", "_hmm", "_hsmm"))]

    for npz_path in npz_files:
        out_path = exp_dir / f"{npz_path.stem}_latent.npz"
        embed_file(npz_path, model, ckpt, out_path)

    print("全部提取完成")


if __name__ == "__main__":
    main()