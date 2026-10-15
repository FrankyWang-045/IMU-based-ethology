"""2 分支频率分解 VAE 训练脚本（支持验证集与验证指标）。"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from model import PoseVAE, kl_loss
from training_dataset import PoseDataset, compute_stats
from utils import ROOT, load_config, output_dir, new_experiment_dir


def get_kl_weights(epoch, kl_betas, kl_ramp_epochs):
    weights = {}
    for name, beta in kl_betas.items():
        ramp = kl_ramp_epochs.get(name, 1)
        if ramp <= 0:
            weights[name] = beta
        else:
            weights[name] = min(1.0, (epoch + 1) / ramp) * beta
    return weights


def split_train_val(all_npz, cfg_dataset, seed):
    """按 recording 划分训练集和验证集。"""
    val_files = set(cfg_dataset.get("val_files") or [])
    val_ratio = cfg_dataset.get("val_ratio", 0.0)

    if val_files:
        train_paths = [p for p in all_npz if p.stem not in val_files]
        val_paths = [p for p in all_npz if p.stem in val_files]
        print(f"使用配置指定的验证集: {[p.stem for p in val_paths]}")
    elif val_ratio and val_ratio > 0:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(all_npz) * val_ratio)))
        val_idx = rng.choice(len(all_npz), size=n_val, replace=False)
        val_idx_set = set(val_idx)
        val_paths = [all_npz[i] for i in val_idx]
        train_paths = [all_npz[i] for i in range(len(all_npz)) if i not in val_idx_set]
        print(f"按比例 {val_ratio:.2f} 随机划分验证集: {[p.stem for p in val_paths]}")
    else:
        train_paths = all_npz
        val_paths = []

    return train_paths, val_paths


@torch.no_grad()
def evaluate(model, loader, device, warmup, rec_w, kl_w):
    """在验证集上计算重建 loss、总 loss 和 R²。"""
    model.eval()

    total_loss_sum = 0.0
    rec_full_sum = 0.0
    rec_slow_sum = 0.0
    rec_fast_sum = 0.0
    kl_sum = {name: 0.0 for name in kl_w}

    # 收集所有预测和真值，用于计算 R²
    all_pred_recon, all_true_recon = [], []
    all_pred_slow, all_true_slow = [], []
    all_pred_fast, all_true_fast = [], []

    n_batches = 0
    for batch in loader:
        x, x_low, x_high = [t.to(device) for t in batch]

        recon_dict, mu, logvar = model.forward_branches(x)

        if warmup > 0:
            pred_recon = recon_dict["recon"][:, warmup:]
            pred_low = recon_dict["slow"][:, warmup:]
            pred_high = recon_dict["fast"][:, warmup:]
            x_ = x[:, warmup:]
            x_low_ = x_low[:, warmup:]
            x_high_ = x_high[:, warmup:]
            mu_ = mu[:, warmup:]
            logvar_ = logvar[:, warmup:]
        else:
            pred_recon = recon_dict["recon"]
            pred_low = recon_dict["slow"]
            pred_high = recon_dict["fast"]
            x_, x_low_, x_high_ = x, x_low, x_high
            mu_, logvar_ = mu, logvar

        rec_full = F.mse_loss(pred_recon, x_)
        rec_slow = F.mse_loss(pred_low, x_low_)
        rec_fast = F.mse_loss(pred_high, x_high_)

        loss = (
            rec_full
            + rec_w.get("slow", 1.0) * rec_slow
            + rec_w.get("fast", 1.0) * rec_fast
        )

        start = 0
        for branch_cfg in model.branch_configs:
            name = branch_cfg["name"]
            zd = branch_cfg["z_dim"]
            kl_b = kl_loss(
                mu_[..., start : start + zd],
                logvar_[..., start : start + zd],
            )
            loss += kl_w[name] * kl_b
            kl_sum[name] += kl_b.item()
            start += zd

        total_loss_sum += loss.item()
        rec_full_sum += rec_full.item()
        rec_slow_sum += rec_slow.item()
        rec_fast_sum += rec_fast.item()
        n_batches += 1

        # 展平 (B, T, C) -> (-1, C) 用于 R²
        all_pred_recon.append(pred_recon.reshape(-1, pred_recon.shape[-1]).cpu())
        all_true_recon.append(x_.reshape(-1, x_.shape[-1]).cpu())
        all_pred_slow.append(pred_low.reshape(-1, pred_low.shape[-1]).cpu())
        all_true_slow.append(x_low_.reshape(-1, x_low_.shape[-1]).cpu())
        all_pred_fast.append(pred_high.reshape(-1, pred_high.shape[-1]).cpu())
        all_true_fast.append(x_high_.reshape(-1, x_high_.shape[-1]).cpu())

    n = max(n_batches, 1)
    metrics = {
        "loss": total_loss_sum / n,
        "rec_full": rec_full_sum / n,
        "rec_slow": rec_slow_sum / n,
        "rec_fast": rec_fast_sum / n,
    }
    for name in kl_sum:
        metrics[f"kl_{name}"] = kl_sum[name] / n

    def _r2(pred, true):
        ss_res = ((pred - true) ** 2).sum(dim=0)
        mean = true.mean(dim=0, keepdim=True)
        ss_tot = ((true - mean) ** 2).sum(dim=0)
        per_dim = 1 - ss_res / (ss_tot + 1e-8)
        return per_dim.mean().item()

    metrics["r2_full"] = _r2(torch.cat(all_pred_recon), torch.cat(all_true_recon))
    metrics["r2_slow"] = _r2(torch.cat(all_pred_slow), torch.cat(all_true_slow))
    metrics["r2_fast"] = _r2(torch.cat(all_pred_fast), torch.cat(all_true_fast))

    return metrics


def plot_val_reconstruction(model, loader, device, warmup, exp_dir, fs=25.0):
    """保存验证集重建可视化：每个轴的 full / slow / high 对比。"""
    model.eval()
    x, x_low, x_high = next(iter(loader))
    x = x[:1].to(device)
    x_low = x_low[:1].to(device)
    x_high = x_high[:1].to(device)

    with torch.no_grad():
        recon_dict, _, _ = model.forward_branches(x)

    if warmup > 0:
        pred_recon = recon_dict["recon"][:, warmup:]
        pred_low = recon_dict["slow"][:, warmup:]
        pred_high = recon_dict["fast"][:, warmup:]
        x_ = x[:, warmup:]
        x_low_ = x_low[:, warmup:]
        x_high_ = x_high[:, warmup:]
    else:
        pred_recon = recon_dict["recon"]
        pred_low = recon_dict["slow"]
        pred_high = recon_dict["fast"]
        x_, x_low_, x_high_ = x, x_low, x_high

    T = x_.shape[1]
    t = np.arange(T) / fs

    axes_names = ["ax", "ay", "az", "wx", "wy", "wz"]
    fig, axes = plt.subplots(6, 3, figsize=(15, 18), sharex=True)

    for c in range(6):
        axes[c, 0].plot(t, x_[0, :, c].cpu(), label="true", alpha=0.7)
        axes[c, 0].plot(t, pred_recon[0, :, c].cpu(), label="recon", alpha=0.7)
        axes[c, 0].set_ylabel(axes_names[c])
        if c == 0:
            axes[c, 0].set_title("full")
            axes[c, 0].legend()

        axes[c, 1].plot(t, x_low_[0, :, c].cpu(), label="true low", alpha=0.7)
        axes[c, 1].plot(t, pred_low[0, :, c].cpu(), label="recon low", alpha=0.7)
        if c == 0:
            axes[c, 1].set_title("slow")
            axes[c, 1].legend()

        axes[c, 2].plot(t, x_high_[0, :, c].cpu(), label="true high", alpha=0.7)
        axes[c, 2].plot(t, pred_high[0, :, c].cpu(), label="recon high", alpha=0.7)
        if c == 0:
            axes[c, 2].set_title("fast")
            axes[c, 2].legend()

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    axes[-1, 2].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(exp_dir / "val_reconstruction.png", dpi=150)
    plt.close(fig)


def main():
    cfg = load_config()
    tc, mc, dc = cfg["train"], cfg["model"], cfg["dataset"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = tc.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("使用设备:", device)

    out = output_dir(cfg)
    all_npz = sorted(out.glob("*.npz"))
    all_npz = [
        p for p in all_npz
        if not p.stem.endswith(("_latent", "_hmm", "_hsmm"))
    ]

    train_paths, val_paths = split_train_val(all_npz, dc, seed)

    if not train_paths:
        raise FileNotFoundError(f"{out} 中没有可用的训练 npz")

    print(f"训练文件: {[p.stem for p in train_paths]}")
    if val_paths:
        print(f"验证文件: {[p.stem for p in val_paths]}")

    mean, std = compute_stats(train_paths)

    aug_cfg = tc.get("augment", {})
    train_datasets = [
        PoseDataset(
            p,
            chunk=dc["chunk"],
            mean=mean,
            std=std,
            lowpass_cutoff=tc.get("lowpass_cutoff", tc.get("slow_lowpass_cutoff", 1.0)),
            fs=cfg["preprocess"]["target_fs"],
            augment=aug_cfg.get("enabled", False),
            aug_max_angle_deg=aug_cfg.get("max_angle_deg", 15.0),
            aug_p=aug_cfg.get("prob", 1.0),
        )
        for p in train_paths
    ]
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=tc["batch_size"],
        shuffle=True,
    )

    val_loader = None
    if val_paths:
        val_datasets = [
            PoseDataset(
                p,
                chunk=dc["chunk"],
                mean=mean,
                std=std,
                lowpass_cutoff=tc.get("lowpass_cutoff", tc.get("slow_lowpass_cutoff", 1.0)),
                fs=cfg["preprocess"]["target_fs"],
                augment=False,
            )
            for p in val_paths
        ]
        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=tc["batch_size"],
            shuffle=False,
        )

    decoder_hidden = mc.get("decoder_hidden", [64, 128])
    branch_configs = mc.get("branches", None)

    model = PoseVAE(
        in_features=mc["in_features"],
        z_dim=mc["z_dim"],
        branch_configs=branch_configs,
        decoder_hidden=decoder_hidden,
        pad_mode="replicate",
    ).to(device)

    warmup = model.receptive_field - 1
    if dc["chunk"] <= warmup:
        raise ValueError(
            f"chunk={dc['chunk']} 必须大于 receptive_field={model.receptive_field}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    rec_w = tc.get("rec_weights", {"slow": 1.0, "fast": 1.0})
    kl_betas = tc.get("kl_betas", {"slow": 1.0e-3, "fast": 1.0e-4})
    kl_ramp_epochs = tc.get("kl_ramp_epochs", {"slow": 10, "fast": 30})
    grad_clip = tc.get("grad_clip", 1.0)

    history = {"train_total": [], "val_total": []}
    for branch_cfg in model.branch_configs:
        history[f"{branch_cfg['name']}_train_rec"] = []
        history[f"{branch_cfg['name']}_val_rec"] = []
        history[f"{branch_cfg['name']}_val_r2"] = []

    best_val_loss = float("inf")

    for epoch in range(tc["epochs"]):
        model.train()
        kl_w = get_kl_weights(epoch, kl_betas, kl_ramp_epochs)

        total_loss = 0.0
        epoch_rec = {branch_cfg["name"]: 0.0 for branch_cfg in model.branch_configs}

        for batch in train_loader:
            x, x_low, x_high = [t.to(device) for t in batch]

            recon_dict, mu, logvar = model.forward_branches(x)

            if warmup > 0:
                pred_recon = recon_dict["recon"][:, warmup:]
                pred_low = recon_dict["slow"][:, warmup:]
                pred_high = recon_dict["fast"][:, warmup:]
                x_ = x[:, warmup:]
                x_low_ = x_low[:, warmup:]
                x_high_ = x_high[:, warmup:]
                mu_ = mu[:, warmup:]
                logvar_ = logvar[:, warmup:]
            else:
                pred_recon = recon_dict["recon"]
                pred_low = recon_dict["slow"]
                pred_high = recon_dict["fast"]
                x_, x_low_, x_high_ = x, x_low, x_high
                mu_, logvar_ = mu, logvar

            rec_full = F.mse_loss(pred_recon, x_)
            rec_slow = F.mse_loss(pred_low, x_low_)
            rec_fast = F.mse_loss(pred_high, x_high_)

            loss = (
                rec_full
                + rec_w.get("slow", 1.0) * rec_slow
                + rec_w.get("fast", 1.0) * rec_fast
            )

            start = 0
            for branch_cfg in model.branch_configs:
                name = branch_cfg["name"]
                zd = branch_cfg["z_dim"]
                kl_b = kl_loss(
                    mu_[..., start : start + zd],
                    logvar_[..., start : start + zd],
                )
                loss += kl_w[name] * kl_b
                start += zd

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            epoch_rec["slow"] += rec_slow.item()
            epoch_rec["fast"] += rec_fast.item()

        n = len(train_loader)
        history["train_total"].append(total_loss / n)
        log_parts = [f"Epoch {epoch+1:3d}/{tc['epochs']}  train={total_loss/n:.4f}"]

        for branch_cfg in model.branch_configs:
            name = branch_cfg["name"]
            history[f"{name}_train_rec"].append(epoch_rec[name] / n)

        # 验证
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, warmup, rec_w, kl_w)
            history["val_total"].append(val_metrics["loss"])
            log_parts.append(f"val={val_metrics['loss']:.4f}")
            log_parts.append(
                f"R² full={val_metrics['r2_full']:.3f} "
                f"slow={val_metrics['r2_slow']:.3f} "
                f"fast={val_metrics['r2_fast']:.3f}"
            )
            for branch_cfg in model.branch_configs:
                name = branch_cfg["name"]
                history[f"{name}_val_rec"].append(val_metrics[f"rec_{name}"])
                history[f"{name}_val_r2"].append(val_metrics[f"r2_{name}"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
        else:
            history["val_total"].append(None)

        print("  ".join(log_parts))

    exp_dir = new_experiment_dir(cfg)

    # 保存最终模型
    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": mean,
            "x_std": std,
            "receptive_field": model.receptive_field,
            "config": {
                "in_features": mc["in_features"],
                "z_dim": mc["z_dim"],
                "branch_configs": model.branch_configs,
                "decoder_hidden": decoder_hidden,
                "pad_mode": model.pad_mode,
            },
        },
        exp_dir / "pose_vae.pt",
    )

    # 保存最佳验证模型
    if val_loader is not None and best_val_loss < float("inf"):
        torch.save(
            {
                "model_state": model.state_dict(),
                "x_mean": mean,
                "x_std": std,
                "receptive_field": model.receptive_field,
                "best_val_loss": best_val_loss,
                "config": {
                    "in_features": mc["in_features"],
                    "z_dim": mc["z_dim"],
                    "branch_configs": model.branch_configs,
                    "decoder_hidden": decoder_hidden,
                    "pad_mode": model.pad_mode,
                },
            },
            exp_dir / "best_pose_vae.pt",
        )
        print(f"最佳验证 loss: {best_val_loss:.4f}")

    shutil.copy(ROOT / "config.yaml", exp_dir / "config.yaml")

    # 保存训练历史
    with open(exp_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # 保存验证集文件列表
    if val_paths:
        with open(exp_dir / "val_files.json", "w") as f:
            json.dump([p.stem for p in val_paths], f, indent=2)

    # 绘制训练/验证曲线
    n_branches = len(model.branch_order)
    fig, axes = plt.subplots(
        2, max(2, n_branches + 1), figsize=(5 * (n_branches + 1), 8), squeeze=False
    )

    axes[0, 0].plot(history["train_total"], label="train")
    val_total = [v for v in history["val_total"] if v is not None]
    if val_total:
        axes[0, 0].plot(
            [i for i, v in enumerate(history["val_total"]) if v is not None],
            val_total,
            label="val",
        )
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    for i, branch_cfg in enumerate(model.branch_configs):
        name = branch_cfg["name"]
        axes[0, i + 1].plot(history[f"{name}_train_rec"], label="train")
        if val_loader is not None:
            axes[0, i + 1].plot(history[f"{name}_val_rec"], label="val")
        axes[0, i + 1].set_title(f"{name} reconstruction")
        axes[0, i + 1].legend()

        axes[1, i + 1].plot(history[f"{name}_val_r2"], label="val R²")
        axes[1, i + 1].set_title(f"{name} validation R²")
        axes[1, i + 1].legend()

    fig.tight_layout()
    fig.savefig(exp_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # 验证集重建可视化
    if val_loader is not None:
        plot_val_reconstruction(
            model, val_loader, device, warmup, exp_dir,
            fs=cfg["preprocess"]["target_fs"],
        )

    print(f"实验产物已保存到 {exp_dir}")


if __name__ == "__main__":
    main()