# -*- coding: utf-8 -*-
"""路线 D 训练 CLI：vameimu.trainer.train_chunk 的薄封装（断点续训）。

用法：venv_python scripts/routeD_train.py <chunk_epochs>
反复运行同一命令直至打印 TRAINING_DONE。
"""
import sys
from pathlib import Path

WORKSPACE = Path(r"F:\Kimi\IMU")
for p in (str(WORKSPACE / "VAME-Style"), str(WORKSPACE / "VAME-IMU")):
    if p not in sys.path:
        sys.path.insert(0, p)

from vame.util.auxiliary import read_config                # noqa: E402
from vameimu import trainer                                # noqa: E402

PROJECT = WORKSPACE / "VAME-Style" / "runs" / "routeD_posture_vae_50hz" / "vame_project"

chunk = int(sys.argv[1]) if len(sys.argv) > 1 else 5
cfg = read_config(str(PROJECT / "config.yaml"))
cfg["model_convergence"] = 10 ** 9      # 禁用原生提前收敛，按 epoch 计数
trainer.train_chunk(cfg, total_epochs=80, chunk_epochs=chunk)
