"""Latent UMAP：左图按个体(文件)着色，右图按 VQ code 着色。

UMAP 在全体采样数据上拟合一次，两图共用坐标系。
每只鼠抽取 5000 点。
"""

import matplotlib.pyplot as plt
import numpy as np
import umap

from utils import load_config, output_dir, latest_experiment_dir

N_PER_FILE = 5000
TOP_CODES = 15          # code 着色图只高亮高频码，其余灰色


def main():
    cfg = load_config()
    out = output_dir(cfg)
    exp_dir = latest_experiment_dir(cfg)

    # ---------- 收集数据 ----------
    zs, codes, owners = [], [], []
    for p in sorted(out.glob("*_latent.npz")):
        d = np.load(p)
        z, c = d["downstream"], d["codes"]
        take = min(N_PER_FILE, len(z))
        idx = np.random.default_rng(0).choice(len(z), take, replace=False)
        zs.append(z[idx]); codes.append(c[idx])
        owners += [p.stem.replace("_latent", "")] * take

    z = np.concatenate(zs)
    codes = np.concatenate(codes)
    owners = np.array(owners)
    print(f"总采样点: {len(z)}")

    # ---------- 一次拟合，两图共用 ----------
    emb = umap.UMAP(n_neighbors=30, min_dist=0.1,
                    random_state=42).fit_transform(z)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    # 左：按个体
    for name in np.unique(owners):
        m = owners == name
        axes[0].scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.4, label=name)
    axes[0].legend(markerscale=5, fontsize=7)
    axes[0].set_title("colored by individual")

    # 右：按 code（高频码彩色，其余灰）
    top = set(np.argsort(np.bincount(codes))[::-1][:TOP_CODES])
    rest = ~np.isin(codes, list(top))
    axes[1].scatter(emb[rest, 0], emb[rest, 1], s=2, alpha=0.15,
                    c="lightgray", label="other")
    for c in sorted(top):
        m = codes == c
        axes[1].scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.5,
                        label=f"code {c}")
    axes[1].legend(markerscale=5, fontsize=7)
    axes[1].set_title(f"colored by code (top {TOP_CODES})")

    fig.tight_layout()
    fig.savefig(exp_dir / "latent_umap.png", dpi=150)
    plt.show()
    print(f"已保存到 {exp_dir / 'latent_umap.png'}")


if __name__ == "__main__":
    main()