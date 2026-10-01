import numpy as np
import torch
from torch.utils.data import DataLoader

from model import PoseVAE, reconstruction_loss, kl_loss
from training_dataset import PoseDataset


# ---------- 配置 ----------
NPZ_PATH = "output/A5_C5_C5-c8.npz"
CHUNK = 1000
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
BETA = 1.0            # KL 损失权重

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备:", device)

# ---------- 数据与模型 ----------
dataset = PoseDataset(NPZ_PATH, chunk=CHUNK)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = PoseVAE(in_features=5, z_dim=6).to(device)   # 模型搬上 GPU
warmup = model.receptive_field - 1                    # 预热帧掩码

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

history = {"rec": [], "kl": [], "total": []}

for epoch in range(EPOCHS):
    model.train()                 # 切换到训练模式
    total_rec, total_kl = 0.0, 0.0

    for batch in loader:
        batch = batch.to(device)                      # 数据搬上 GPU

        x_recon, mu, logvar = model.forward(batch)
        rec = reconstruction_loss(x_recon[:, warmup:], batch[:, warmup:])
        kl  = kl_loss(mu[:, warmup:], logvar[:, warmup:])
        loss = rec + BETA * kl

        optimizer.zero_grad()                          # 清梯度
        loss.backward()                                # 反向传播
        optimizer.step()                               # 更新参数


        total_rec += rec.item()
        total_kl += kl.item()
    
    n = len(loader)
    avg_rec, avg_kl = total_rec / n, total_kl / n

    history["rec"].append(avg_rec)              # 新增三行
    history["kl"].append(avg_kl)
    history["total"].append(avg_rec + BETA * avg_kl)

    print(f"Epoch {epoch+1:3d}/{EPOCHS}  rec={avg_rec:.4f}  kl={avg_kl:.4f}")
    n = len(loader)
    print(f"Epoch {epoch+1:3d}/{EPOCHS}  rec={total_rec/n:.4f}  kl={total_kl/n:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "x_mean": dataset.x_mean,          # ← 替换掉 wz_mean/wz_std
    "x_std": dataset.x_std,
    "receptive_field": model.receptive_field,
    "config": {"in_features": 5, "z_dim": 6},
}, "output/pose_vae.pt")


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["rec"], label="reconstruction")
axes[0].plot(history["kl"], label="KL")
axes[0].plot(history["total"], label="total", linestyle="--")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss curves")
axes[0].legend()

axes[1].plot(history["kl"], color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_title("KL (zoomed)")
axes[1].set_yscale("log")      # KL 通常很小，log 坐标看得更清楚

plt.tight_layout()
plt.savefig("output/loss_curve.png", dpi=150)
plt.show()