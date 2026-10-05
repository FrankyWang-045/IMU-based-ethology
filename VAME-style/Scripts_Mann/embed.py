"""批量 latent 提取：对所有预处理 npz，用指定实验的模型生成下游特征。

用法：
    python embed.py            # 默认使用最新的实验目录
    python embed.py exp_001    # 指定实验目录
"""

import sys

import numpy as np
import torch

from utils import load_config, output_dir, latent_path, latest_experiment_dir
from model import PoseVQVAE


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(exp_dir, device):
    ckpt = torch.load(exp_dir / "pose_vae.pt",
                      map_location=device, weights_only=False)
    model = PoseVQVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def build_vae_input(feats, x_mean, x_std):
    """从 features 矩阵构造 VAE 输入，用训练时的统计量标准化。"""
    trig = feats[:, 9:13]
    wz = feats[:, 5:6]
    x_raw = np.hstack([trig, wz])
    return ((x_raw - x_mean) / x_std).astype(np.float32)


def extract_latent(model, x, device):
    """整段前向，eval 模式下取 mu 作为每帧 latent。"""
def extract_latent(model, x, device):
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        _, z_q, codes, _ = model(x_t)
    return z_q[0].cpu().numpy(), codes[0].cpu().numpy()


def build_bypass(feats):
    """旁路特征接口（暂未实现）。返回 None 时跳过拼接。"""
    return None


def embed_file(npz_path, model, ckpt, out_path):
    """处理单个文件：npz -> latent npz。"""
    data = np.load(npz_path)
    feats = data["features"]

    x = build_vae_input(feats, ckpt["x_mean"], ckpt["x_std"])
    z, codes = extract_latent(model, x, device)

    bypass = build_bypass(feats)
    downstream = z if bypass is None else np.hstack([z, bypass])

    warmup = model.receptive_field - 1
    downstream = downstream[warmup:].astype(np.float32)
    time = data["time"][warmup:]
    interp_mask = data["interp_mask"][warmup:]
    assert len(downstream) == len(time) == len(interp_mask), "长度对齐失败"

    np.savez_compressed(out_path, downstream=downstream,
                        codes=codes[warmup:],
                        time=time, interp_mask=interp_mask)

def main():
    cfg = load_config()
    exp_dir = (output_dir(cfg) / sys.argv[1]) if len(sys.argv) > 1 \
        else latest_experiment_dir(cfg)
    print(f"使用模型: {exp_dir}")

    model, ckpt = load_model(exp_dir, device)
    npz_files = [p for p in sorted(output_dir(cfg).glob("*.npz"))
                 if not p.stem.endswith(("_latent", "_hmm"))]

    for npz_path in npz_files:
        embed_file(npz_path, model, ckpt, latent_path(cfg, npz_path.stem))
    print("全部提取完成")


if __name__ == "__main__":
    main()