from pathlib import Path

import numpy as np
import torch

from model_ae import PoseAE
from utils import load_config, output_dir


@torch.no_grad()
def extract_latent(cfg, chunk_size=10000):
    out = output_dir(cfg)
    data_dir = out / "direction_gravity"
    exp_dir = out / "exp_ae"
    model_path = exp_dir / "pose_ae.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    mean = checkpoint["x_mean"]
    std = checkpoint["x_std"]
    mc = checkpoint["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseAE(
        in_features=mc["in_features"],
        z_dim=mc["z_dim"],
        cnn_channels=mc["cnn_channels"],
        kernel_sizes=mc["kernel_sizes"],
        gru_hidden=mc["gru_hidden"],
        gru_layers=mc["gru_layers"],
        decoder_hidden=mc["decoder_hidden"],
        dynamics_hidden=mc["dynamics_hidden"],
        pad_mode=mc["pad_mode"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    latent_dir = out / "latent"
    latent_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in sorted(data_dir.glob("*.npz")):
        data = np.load(npz_path)
        x = (data["features"].astype(np.float32) - mean) / (std + 1e-8)

        # 对长序列分 chunk 提取，避免显存/ cuDNN 问题
        if len(x) <= chunk_size:
            x_t = torch.from_numpy(x).unsqueeze(0).to(device).contiguous()
            z = model.encode(x_t)[0].cpu().numpy()
        else:
            z_chunks = []
            for start in range(0, len(x), chunk_size):
                x_chunk = x[start : start + chunk_size]
                x_t = torch.from_numpy(x_chunk).unsqueeze(0).to(device).contiguous()
                z_chunk = model.encode(x_t)[0].cpu().numpy()
                z_chunks.append(z_chunk)
            z = np.concatenate(z_chunks, axis=0)

        np.savez_compressed(
            latent_dir / f"{npz_path.stem}_latent.npz",
            latent=z,
            time=data["time"],
            interp_mask=data["interp_mask"],
        )
        print(f"  已保存 latent: {latent_dir / npz_path.stem}_latent.npz, shape={z.shape}")

    print(f"全部 latent 已保存到 {latent_dir}")


if __name__ == "__main__":
    cfg = load_config()
    extract_latent(cfg)