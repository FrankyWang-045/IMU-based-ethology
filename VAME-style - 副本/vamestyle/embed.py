# -*- coding: utf-8 -*-
"""embed：用冻结的 VAME 编码器对全部 session 生成 mu latent (T-29, 16)。

落盘两处（幂等）：
  runs/.../results/<session>/VAME/latent_vectors.npy（工程口径）
  cache/mu_<session>.npy（v4.0 缓存，HMM 管线只读这里）

用法：python -m vamestyle.embed
"""
import numpy as np

from vamestyle.dataset import CFG, ROOT, sessions


def main():
    from vame.util.auxiliary import read_config
    from vameimu.vame_pipeline import embed

    cfg = read_config(CFG["vame"]["project_dir"] + "/config.yaml")
    lat = embed(cfg, batch_size=256)
    out = ROOT / "cache"
    out.mkdir(exist_ok=True)
    for s in sessions():
        mu = lat[s.name]
        assert mu.shape[1] == CFG["vame"]["zdims"], mu.shape
        assert len(mu) == s.T - (CFG["vame"]["time_window"] - 1), (len(mu), s.T)
        np.save(out / f"mu_{s.name}.npy", mu.astype(np.float32))
        print(f"[embed] {s.name}: mu {mu.shape} -> cache/mu_{s.name}.npy")


if __name__ == "__main__":
    main()
