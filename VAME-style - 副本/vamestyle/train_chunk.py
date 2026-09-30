# -*- coding: utf-8 -*-
"""v4.0 分块续训驱动（前台 ≤300 s 一块，幂等，中断后重跑本文件即可）。

用法：python -m vamestyle.train_chunk
"""
import torch

# chunk_state.pkl 内含 ruamel ScalarFloat（vame config 值），torch 2.6
# weights_only 默认拒绝；自家检查点，回退 weights_only=False。
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

from vame.util.auxiliary import read_config
from vameimu.trainer import train_chunk
from vamestyle.dataset import CFG

if __name__ == "__main__":
    cfg = read_config(CFG["vame"]["project_dir"] + "/config.yaml")
    train_chunk(cfg, total_epochs=CFG["vame"]["total_epochs"],
                chunk_epochs=CFG["vame"]["epochs_per_chunk"])
