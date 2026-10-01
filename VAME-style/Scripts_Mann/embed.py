"""全序列 latent 提取脚本。

流程：加载 npz -> 用 checkpoint 统计量标准化 -> 整段前向 -> 裁剪预热帧
      -> (预留旁路接口) -> 保存下游特征 npz。
"""

import numpy as np
import torch

from model import PoseVAE

NPZ_PATH = "output/A5_C5_C5-c8.npz"
CKPT_PATH = "output/pose_vae.pt"
OUT_PATH = "output/A5_C5_C5-c8_latent.npz"

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(ckpt.keys())
    model = PoseVAE(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def build_vae_input(feats, x_mean, x_std):
    """从 features 矩阵构造 VAE 输入，用训练时的统计量标准化。"""
    trig = feats[:, 9:13]        # sin/cos 四轴
    wz = feats[:, 5:6]           # gyro z
    x_raw = np.hstack([trig, wz])
    x_stdized = (x_raw - x_mean) / x_std
    return x_stdized.astype(np.float32)


def extract_latent(model, x, device):
    """整段前向，eval 模式下取 mu 作为每帧 latent。"""
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)   # (1, N, 5)
    with torch.no_grad():
        _, mu, _ = model(x_t)
    return mu[0].cpu().numpy()                          # (N, z_dim)


def build_bypass(feats):
    """旁路特征接口（暂未实现）。

    计划：lin_acc(3) + wx, wy(2)，z-score 后与 latent 拼接。
    返回形状应为 (N, n_bypass)。
    """
    return None


def main():
    model, ckpt = load_model(CKPT_PATH, device)
    warmup = ckpt["receptive_field"] - 1

    data = np.load(NPZ_PATH)
    feats = data["features"]

    x = build_vae_input(feats, ckpt["x_mean"], ckpt["x_std"])
    z = extract_latent(model, x, device)

    bypass = build_bypass(feats)
    if bypass is not None:
        downstream = np.hstack([z, bypass])
    else:
        downstream = z

    # 裁剪预热帧：latent、时间戳、掩码必须同步，保证逐帧对齐
    downstream = downstream[warmup:].astype(np.float32)
    time = data["time"][warmup:]
    interp_mask = data["interp_mask"][warmup:]

    assert len(downstream) == len(time) == len(interp_mask), "长度对齐失败"

    np.savez_compressed(
        OUT_PATH,
        downstream=downstream,
        time=time,
        interp_mask=interp_mask,
    )
    print(f"下游特征: {downstream.shape}  时间点数: {len(time)}  已保存到 {OUT_PATH}")


if __name__ == "__main__":
    main()