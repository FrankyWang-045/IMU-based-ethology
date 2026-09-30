# -*- coding: utf-8 -*-
"""无头冒烟：顺序执行 manual_train.ipynb 的 code cells（小参数）。

用法：python tools/smoke_manual_nb.py
覆盖：2 条 session、2 epochs、K=[12]、小拟合块；产物落 runs/_smoke_nb/
（幂等：重跑自动复用已建数据/续训状态）。
"""
import json
import sys
import time
import traceback
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "manual_train.ipynb"

cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
g = {"__name__": "__main__"}
t0 = time.time()
for i, c in enumerate(cells):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    head = src.strip().splitlines()[0][:60]
    print(f"\n===== cell {i}: {head}  [{time.time() - t0:.0f}s]", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), g)
    except Exception:
        traceback.print_exc()
        print(f"SMOKE_FAIL at cell {i}")
        sys.exit(1)
    if "==== §1 参数面板" in src:
        # 小参数覆盖 + 重应用路径
        g["P"]["run_dir"] = str(Path(g["WORKSPACE"]) / "VAME-Style" / "runs" / "_smoke_nb")
        g["P"]["data"]["sessions"] = ["rec_000", "rec_001"]
        g["P"]["data"]["rebuild"] = True
        g["P"]["vae"]["max_epochs"] = 2
        g["P"]["vae"]["epochs_per_block"] = 2
        g["P"]["hmm"]["k_list"] = [12]
        g["P"]["hmm"]["n_sub"] = 15000
        g["P"]["hmm"]["n_iter"] = 10
        g["apply_params"]()
        print("（冒烟覆盖：2 sessions / 2 epochs / K=12 / n_sub=15k）")
print(f"\nSMOKE_OK  总耗时 {time.time() - t0:.0f}s  RUN_DIR={g['P']['run_dir']}")
