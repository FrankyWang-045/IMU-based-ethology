from pathlib import Path

import numpy as np

from utils import load_config, output_dir


def compute_linear_acc_direction(linear_acc, eps=1e-8):
    """
    计算线性加速度方向（单位向量）。
    模长过小的帧方向置为 0，避免除零。
    """
    norms = np.linalg.norm(linear_acc, axis=1, keepdims=True)
    direction = np.zeros_like(linear_acc)
    valid = norms[:, 0] > eps
    direction[valid] = linear_acc[valid] / norms[valid]
    return direction


def merge_direction_and_gravity(
    cfg, input_subdir="", output_subdir="direction_gravity"
):
    """
    读取 9 轴 npz，提取 linear_acc(0:3) 和 gravity(6:9)，
    生成 6 轴新文件：[linear_acc_direction(3), gravity(3)]。
    """
    out = output_dir(cfg)
    in_dir = out / input_subdir if input_subdir else out
    out_dir = out / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"未找到输入 npz: {in_dir}")

    for path in npz_files:
        data = np.load(path)
        features = data["features"]

        if features.shape[1] < 9:
            raise ValueError(
                f"{path.name}: 需要 9 轴输入 [linear_acc(3), gyro(3), gravity(3)]"
            )

        linear_acc = features[:, :3]
        gravity = features[:, 6:9]

        direction = compute_linear_acc_direction(linear_acc)
        merged = np.hstack([direction, gravity]).astype(np.float32)

        np.savez_compressed(
            out_dir / f"{path.stem}.npz",
            features=merged,
            time=data["time"],
            interp_mask=data["interp_mask"],
        )
        print(f"  已保存: {out_dir / path.stem}.npz, shape={merged.shape}")

    print(f"全部完成，输出目录: {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    merge_direction_and_gravity(cfg)