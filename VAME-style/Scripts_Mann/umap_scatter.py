"""Latent 空间可视化：UMAP 降维 + HMM 状态着色。"""

import numpy as np
import matplotlib.pyplot as plt
import umap

LATENT_PATH = "output/A5_C5_C5-c8_latent.npz"
HMM_PATH = "output/A5_C5_C5-c8_hmm.npz"
OUT_FIG = "output/latent_umap.png"
N_SAMPLES = 20000        # UMAP 对 18 万点较慢，先随机抽样

# ---------- 加载 ----------
lat = np.load(LATENT_PATH)
hmm = np.load(HMM_PATH)
z = lat["downstream"]          # (T, 6)
states = hmm["states"]         # (T,)

# ---------- 随机抽样（两数组用同一索引，保持对应） ----------
rng = np.random.default_rng(0)
idx = rng.choice(len(z), size=min(N_SAMPLES, len(z)), replace=False)
z_sub = z[idx]
s_sub = states[idx]

# ---------- UMAP 降维 ----------
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(z_sub)      # (N_SAMPLES, 2)

# ---------- 绘图 ----------
fig, ax = plt.subplots(figsize=(8, 7))

for s in np.unique(s_sub):
    m = s_sub == s
    ax.scatter(embedding[m, 0], embedding[m, 1],
               s=2, alpha=0.5, label=f"state {s}")

ax.legend(markerscale=5, loc="upper right")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("Latent space colored by HMM state")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150)
plt.show()