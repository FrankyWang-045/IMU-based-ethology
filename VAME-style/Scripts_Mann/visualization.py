import numpy as np
import torch
import matplotlib.pyplot as plt

from model import PoseVAE
from training_dataset import PoseDataset

# 1. 加载模型
ckpt = torch.load("output/pose_vae.pt", weights_only=False)
model = PoseVAE(**ckpt["config"])
model.load_state_dict(ckpt["model_state"])
model.eval()

# 2. 准备一段连续数据（不要随机裁剪！取序列中间 2000 帧）
ds = PoseDataset("output/A5_C5_C5-c8.npz")
seg = torch.from_numpy(ds.x[50000:52000]).unsqueeze(0)  # (1, 2000, 5)

# 3. 前向
with torch.no_grad():
    x_recon, mu, logvar = model(seg)

w = model.receptive_field - 1
x = seg[0, w:].numpy()
recon = x_recon[0, w:].numpy()
z = mu[0, w:].numpy()          # eval 模式下 Lambda 返回 mu 作为 z

# 4. 画图
fig, axes = plt.subplots(5 + 1, 1, figsize=(14, 12), sharex=True)
names = ["sin_roll", "cos_roll", "sin_pitch", "cos_pitch", "wz"]
for i, ax in enumerate(axes[:5]):
    ax.plot(x[:, i], label="true", alpha=0.8)
    ax.plot(recon[:, i], label="recon", alpha=0.8)
    ax.set_ylabel(names[i])
    ax.legend(loc="upper right")
for d in range(z.shape[1]):    # 6 维 latent 画在一起
    axes[5].plot(z[:, d], alpha=0.7)
axes[5].set_ylabel("latent z")
plt.tight_layout()
plt.savefig("output/eval_recon.png", dpi=150)
plt.show()