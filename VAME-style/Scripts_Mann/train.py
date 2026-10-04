"""PoseVAE 训练脚本：读 config，支持多文件训练，产物存入实验目录。"""

import shutil

import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader

from model import PoseVAE, reconstruction_loss, kl_loss
from training_dataset import PoseDataset, compute_stats
from utils import ROOT, load_config, output_dir, new_experiment_dir


def main():
    cfg = load_config()
    tc, mc, dc = cfg["train"], cfg["model"], cfg["dataset"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc["seed"])
    print("使用设备:", device)

    # ---------- 数据 ----------
    out = output_dir(cfg)
    val_files = set(dc["val_files"] or [])
    all_npz = sorted(out.glob("*.npz"))
    # 只取预处理产物（排除 *_latent / *_hmm 等结果文件）
    all_npz = [p for p in all_npz if "_" not in p.stem.split("-")[-1]
               or not p.stem.endswith(("_latent", "_hmm"))]
    all_npz = [p for p in all_npz if not p.stem.endswith(("_latent", "_hmm"))]

    train_paths = [p for p in all_npz if p.stem not in val_files]
    if not train_paths:
        raise FileNotFoundError(f"{out} 中没有可用的预处理 npz，请先运行 preprocess.py")

    mean, std = compute_stats(train_paths)
    datasets = [PoseDataset(p, chunk=dc["chunk"], mean=mean, std=std)
                for p in train_paths]
    loader = DataLoader(ConcatDataset(datasets),
                        batch_size=tc["batch_size"], shuffle=True)
    print(f"训练文件: {[p.stem for p in train_paths]}")
    print(f"验证文件: {sorted(val_files) or '无'}")

    # ---------- 模型 ----------
    model = PoseVAE(in_features=mc["in_features"], z_dim=mc["z_dim"]).to(device)
    warmup = model.receptive_field - 1
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    # ---------- 训练 ----------
    history = {"rec": [], "kl": [], "total": []}
    for epoch in range(tc["epochs"]):
        model.train()
        total_rec, total_kl = 0.0, 0.0

        #batch内循环
        for batch in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)

            #计算损失函数
            rec = reconstruction_loss(x_recon[:, warmup:], batch[:, warmup:])
            kl = kl_loss(mu[:, warmup:], logvar[:, warmup:])
            loss = rec + tc["kl_beta"] * kl
            
            #训练模型固定三步
            optimizer.zero_grad()    #梯度清零
            loss.backward()          #反向传播
            optimizer.step()         #更新参数

            #累计损失
            total_rec += rec.item()    
            total_kl += kl.item()

        n = len(loader)
        avg_rec, avg_kl = total_rec / n, total_kl / n
        history["rec"].append(avg_rec)
        history["kl"].append(avg_kl)
        history["total"].append(avg_rec + tc["kl_beta"] * avg_kl)
        print(f"Epoch {epoch+1:3d}/{tc['epochs']}  rec={avg_rec:.4f}  kl={avg_kl:.4f}")

    # ---------- 保存到实验目录 ----------
    exp_dir = new_experiment_dir(cfg)
    torch.save({
        "model_state": model.state_dict(),
        "x_mean": mean,
        "x_std": std,
        "receptive_field": model.receptive_field,
        "config": {"in_features": mc["in_features"], "z_dim": mc["z_dim"]},
    }, exp_dir / "pose_vae.pt")
    shutil.copy(ROOT / "config.yaml", exp_dir / "config.yaml")   # config 副本，保证可复现

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["rec"], label="reconstruction")
    axes[0].plot(history["kl"], label="KL")
    axes[0].plot(history["total"], label="total", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(history["kl"], color="orange")
    axes[1].set_xlabel("Epoch"); axes[1].set_yscale("log"); axes[1].set_title("KL (zoomed)")
    fig.tight_layout()
    fig.savefig(exp_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    print(f"实验产物已保存到 {exp_dir}")


if __name__ == "__main__":
    main()