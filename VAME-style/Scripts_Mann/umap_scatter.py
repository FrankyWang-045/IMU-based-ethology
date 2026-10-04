"""Latent UMAP 降维 + HMM 状态着色。"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import umap

# Compatibility shim: umap-learn 0.5.7 passes force_all_finite to
# sklearn check_array, renamed to ensure_all_finite in scikit-learn >= 1.6.
import umap.umap_ as _umap_mod

_orig_check_array = _umap_mod.check_array


def _check_array_compat(*args, **kwargs):
    if "force_all_finite" in kwargs:
        kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
    return _orig_check_array(*args, **kwargs)


_umap_mod.check_array = _check_array_compat

from utils import load_config, output_dir, latent_path, hmm_path

N_SAMPLES = 20000


def main():
    cfg = load_config()
    out = output_dir(cfg)
    rec = sys.argv[1] if len(sys.argv) > 1 else None
    if rec is None:
        rec = sorted(out.glob("*_hmm.npz"))[0].stem.replace("_hmm", "")

    z = np.load(latent_path(cfg, rec))["downstream"]
    states = np.load(hmm_path(cfg, rec))["states"]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(z), size=min(N_SAMPLES, len(z)), replace=False)
    z_sub, s_sub = z[idx], states[idx]

    embedding = umap.UMAP(n_neighbors=30, min_dist=0.1,
                          random_state=42).fit_transform(z_sub)

    fig, ax = plt.subplots(figsize=(8, 7))
    for s in np.unique(s_sub):
        m = s_sub == s
        ax.scatter(embedding[m, 0], embedding[m, 1],
                   s=2, alpha=0.5, label=f"state {s}")
    ax.legend(markerscale=5, loc="upper right")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(f"{rec} latent space")
    fig.tight_layout()
    fig.savefig(out / f"{rec}_umap.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()