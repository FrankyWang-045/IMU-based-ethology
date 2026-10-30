import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset_ae import PoseAEDataset, compute_stats
from model_lra import PoseLRA, lra_loss
from utils import ROOT, load_config, output_dir


def split_train_val(all_npz, cfg_train, seed):
    val_files = set(cfg_train.get("val_files") or [])
    val_ratio = cfg_train.get("val_ratio", 0.0)

    if val_files:
        train_paths = [p for p in all_npz if p.stem not in val_files]
        val_paths = [p for p in all_npz if p.stem in val_files]
    elif val_ratio and val_ratio > 0:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(all_npz) * val_ratio)))
        val_idx = rng.choice(len(all_npz), size=n_val, replace=False)
        val_idx_set = set(val_idx)
        val_paths = [all_npz[i] for i in val_idx]
        train_paths = [all_npz[i] for i in range(len(all_npz)) if i not in val_idx_set]
    else:
        train_paths = all_npz
        val_paths = []

    return train_paths, val_paths


@torch.no_grad()
def evaluate(model, loader, device, smooth_weight, dyn_weight, warmup, A_reg=0.001):
    model.eval()
    total_loss = 0.0
    rec_sum = smooth_sum = dyn_sum = 0.0
    n = 0

    for batch in loader:
        x = batch.to(device)
        recon, z, z_pred = model(x)

        if warmup > 0:
            recon = recon[:, warmup:]
            x_ = x[:, warmup:]
            z = z[:, warmup:]
            z_pred = z_pred[:, warmup:]
        else:
            x_ = x

        loss, metrics = lra_loss(
            recon, x_, z, z_pred,
            smooth_weight=smooth_weight,
            dyn_weight=dyn_weight,
            A_reg=A_reg,
            A=model.recurrence.A,
        )

        total_loss += loss.item()
        rec_sum += metrics["rec"]
        smooth_sum += metrics["smooth"]
        dyn_sum += metrics["dyn"]
        n += 1

    n = max(n, 1)
    return {
        "loss": total_loss / n,
        "rec": rec_sum / n,
        "smooth": smooth_sum / n,
        "dyn": dyn_sum / n,
    }


def main():
    cfg = load_config()
    mc = cfg["model"]
    tc = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])

    out = output_dir(cfg)
    data_dir = out / "direction_gravity"
    all_npz = sorted(data_dir.glob("*.npz"))
    if not all_npz:
        raise FileNotFoundError(f"{data_dir} 中没有 npz 文件")

    train_paths, val_paths = split_train_val(all_npz, tc, tc["seed"])
    print(f"训练集: {[p.stem for p in train_paths]}")
    print(f"验证集: {[p.stem for p in val_paths]}")

    mean, std = compute_stats(train_paths)

    train_datasets = [PoseAEDataset(p, chunk=tc["chunk"], mean=mean, std=std) for p in train_paths]
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=tc["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val_loader = None
    if val_paths:
        val_datasets = [PoseAEDataset(p, chunk=tc["chunk"], mean=mean, std=std) for p in val_paths]
        val_loader = DataLoader(
            ConcatDataset(val_datasets),
            batch_size=tc["batch_size"],
            shuffle=False,
        )

    model = PoseLRA(
        in_features=mc["in_features"],
        z_dim=mc["z_dim"],
        cnn_channels=mc["cnn_channels"],
        kernel_sizes=mc["kernel_sizes"],
        decoder_hidden=mc["decoder_hidden"],
        pad_mode=mc["pad_mode"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    warmup = model.receptive_field
    A_reg = tc.get("A_reg", 0.001)

    history = {
        "train_loss": [], "train_rec": [], "train_smooth": [], "train_dyn": [],
        "val_loss": [], "val_rec": [], "val_smooth": [], "val_dyn": [],
    }
    best_val_loss = float("inf")

    for epoch in range(tc["epochs"]):
        model.train()
        epoch_loss = epoch_rec = epoch_smooth = epoch_dyn = 0.0

        for batch in train_loader:
            x = batch.to(device)
            recon, z, z_pred = model(x)

            if warmup > 0:
                recon = recon[:, warmup:]
                x_ = x[:, warmup:]
                z = z[:, warmup:]
                z_pred = z_pred[:, warmup:]
            else:
                x_ = x

            loss, metrics = lra_loss(
                recon, x_, z, z_pred,
                smooth_weight=tc["smooth_weight"],
                dyn_weight=tc["dyn_weight"],
                A_reg=A_reg,
                A=model.recurrence.A,
            )

            optimizer.zero_grad()
            loss.backward()
            if tc.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec += metrics["rec"]
            epoch_smooth += metrics["smooth"]
            epoch_dyn += metrics["dyn"]

        n = len(train_loader)
        history["train_loss"].append(epoch_loss / n)
        history["train_rec"].append(epoch_rec / n)
        history["train_smooth"].append(epoch_smooth / n)
        history["train_dyn"].append(epoch_dyn / n)

        log_parts = [f"Epoch {epoch+1:3d}/{tc['epochs']}  train={epoch_loss/n:.4f}"]

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, tc["smooth_weight"], tc["dyn_weight"], warmup, A_reg)
            for k in ["loss", "rec", "smooth", "dyn"]:
                history[f"val_{k}"].append(val_metrics[k])
            log_parts.append(f"val={val_metrics['loss']:.4f}")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
        else:
            for k in ["loss", "rec", "smooth", "dyn"]:
                history[f"val_{k}"].append(None)

        print("  ".join(log_parts))

    # 保存结果
    exp_dir = out / "exp_lra"
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "x_mean": mean,
            "x_std": std,
            "receptive_field": model.receptive_field,
            "config": mc,
        },
        exp_dir / "pose_lra.pt",
    )

    if val_paths:
        with open(exp_dir / "val_files.json", "w") as f:
            json.dump([p.stem for p in val_paths], f, indent=2)

    with open(exp_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    shutil.copy(ROOT / "config.yaml", exp_dir / "config.yaml")

    # 训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes.flatten(), ["loss", "rec", "smooth", "dyn"]):
        ax.plot(history[f"train_{name}"], label="train")
        val_data = [v for v in history[f"val_{name}"] if v is not None]
        if val_data:
            idx = [i for i, v in enumerate(history[f"val_{name}"]) if v is not None]
            ax.plot(idx, val_data, label="val")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(exp_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    print(f"实验结果已保存到 {exp_dir}")


if __name__ == "__main__":
    main()