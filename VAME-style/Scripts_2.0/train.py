"""2 分支频率分解 VAE 训练脚本。"""

import shutil

import matplotlib.pyplot as plt
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


def main():
    cfg = load_config()
    tc, mc, dc = cfg["train"], cfg["model"], cfg["dataset"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tc["seed"])
    print("使用设备:", device)

    out = output_dir(cfg)
    val_files = set(dc.get("val_files") or [])
    all_npz = sorted(out.glob("*.npz"))
    all_npz = [
        p for p in all_npz
        if not p.stem.endswith(("_latent", "_hmm", "_hsmm"))
    ]

    train_paths = [p for p in all_npz if p.stem not in val_files]
    if not train_paths:
        raise FileNotFoundError(f"{out} 中没有可用的预处理 npz")

    mean, std = compute_stats(train_paths)
    datasets = [
        PoseDataset(
            p,
            chunk=dc["chunk"],
            mean=mean,
            std=std,
            lowpass_cutoff=tc.get("lowpass_cutoff", tc.get("slow_lowpass_cutoff", 1.0)),
            fs=cfg["preprocess"]["target_fs"],
        )
        for p in train_paths
    ]
    loader = DataLoader(
        ConcatDataset(datasets),
        batch_size=tc["batch_size"],
        shuffle=True,
    )
    print(f"训练文件: {[p.stem for p in train_paths]}")

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

    history = {"total": []}
    for branch_cfg in model.branch_configs:
        history[f"{branch_cfg['name']}_rec"] = []
        history[f"{branch_cfg['name']}_kl"] = []

    for epoch in range(tc["epochs"]):
        model.train()
        kl_w = get_kl_weights(epoch, kl_betas, kl_ramp_epochs)

        total_loss = 0.0
        epoch_rec = {branch_cfg["name"]: 0.0 for branch_cfg in model.branch_configs}
        epoch_kl = {branch_cfg["name"]: 0.0 for branch_cfg in model.branch_configs}

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
                epoch_kl[name] += kl_b.item()
                start += zd

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            epoch_rec["slow"] += rec_slow.item()
            epoch_rec["fast"] += rec_fast.item()

        n = len(loader)
        history["total"].append(total_loss / n)
        log_parts = [f"Epoch {epoch+1:3d}/{tc['epochs']}  total={total_loss/n:.4f}"]
        for branch_cfg in model.branch_configs:
            name = branch_cfg["name"]
            history[f"{name}_rec"].append(epoch_rec[name] / n)
            history[f"{name}_kl"].append(epoch_kl[name] / n)
            log_parts.append(
                f"{name}: rec={epoch_rec[name]/n:.4f} kl={epoch_kl[name]/n:.4f} "
                f"(β={kl_w[name]:.2e})"
            )
        print("  ".join(log_parts))

    exp_dir = new_experiment_dir(cfg)
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
    shutil.copy(ROOT / "config.yaml", exp_dir / "config.yaml")

    n_branches = len(model.branch_order)
    fig, axes = plt.subplots(
        2, max(2, n_branches), figsize=(5 * max(2, n_branches), 8), squeeze=False
    )

    axes[0, 0].plot(history["total"])
    axes[0, 0].set_title("Total Loss")

    for i, branch_cfg in enumerate(model.branch_configs):
        name = branch_cfg["name"]
        axes[0, i].plot(history[f"{name}_rec"])
        axes[0, i].set_title(f"{name} reconstruction")
        axes[1, i].plot(history[f"{name}_kl"])
        axes[1, i].set_title(f"{name} KL")
        axes[1, i].set_yscale("log")

    fig.tight_layout()
    fig.savefig(exp_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    print(f"实验产物已保存到 {exp_dir}")


if __name__ == "__main__":
    main()