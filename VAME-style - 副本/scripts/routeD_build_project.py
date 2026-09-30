# -*- coding: utf-8 -*-
"""路线 D：构建姿态-VAE 工程（50 Hz，pilot 3 条，grav3 输入）。

VAE 输入 = grav3（feat9 3:6 单位向量）逐 session robust-z；
3 通道 nc（keypoints=["grav"] × xyz），config 由 vame_25hz_14s 克隆并补丁：
num_features=3、session_names=3 条、路径/工程名；其余超参与现行版完全一致
（time_window=30, zdims=16, pred_steps=15, GRU, beta=1.0, lr 5e-4, batch 256）。
幂等：nc/config 已存在则跳过，create_trainset 重复调用安全。
用法：venv_python scripts/routeD_build_project.py
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

WORKSPACE = Path(r"F:\Kimi\IMU")
VAME_STYLE = WORKSPACE / "VAME-Style"
VAME_IMU = WORKSPACE / "VAME-IMU"
for p in (str(VAME_STYLE), str(VAME_IMU)):
    if p not in sys.path:
        sys.path.insert(0, p)

from vameimu.data import robust_z                                        # noqa: E402
from vame.util.auxiliary import read_config, write_config                # noqa: E402

SESS = ["rec_000", "rec_005", "rec_010"]
DS50 = VAME_IMU / "data" / "ds50"
PROJECT = VAME_STYLE / "runs" / "routeD_posture_vae_50hz" / "vame_project"
TEMPLATE = VAME_STYLE / "runs" / "vame_25hz_14s" / "vame_project" / "config.yaml"
FS = 50.0


def write_grav_nc(out_path, g3):
    """(T,3) 姿态通道 → VAME processed nc（keypoints=['grav'] × xyz）。"""
    T = len(g3)
    dims = ("time", "space", "keypoints", "individuals")
    vals = g3.T.reshape(3, 1, T, 1).transpose(2, 0, 1, 3)
    ds = xr.Dataset(
        {"position": (dims, vals),
         "position_processed": (dims, vals),
         "confidence": (("time", "keypoints", "individuals"),
                        np.ones((T, 1, 1), dtype=np.float32))},
        coords={"time": np.arange(T, dtype=np.float64),
                "space": ["x", "y", "z"],
                "keypoints": ["grav"],
                "individuals": ["mouse0"]},
        attrs={"fps": FS, "source_software": "vameimu",
               "centered_reference_keypoint": "__none__",
               "orientation_reference_keypoint": "__none__"},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(str(out_path))


def main():
    for s in SESS:
        out = PROJECT / "data" / "processed" / f"{s}_processed.nc"
        if out.exists():
            print(f"[data] {s}: nc 已存在，跳过")
            continue
        d = np.load(DS50 / f"{s}.npz")
        g3 = robust_z(d["feat9"][:, 3:6].astype(np.float64), use_iqr=True)
        assert np.isfinite(g3).all()
        write_grav_nc(out, g3)
        print(f"[data] {s}: grav nc ({len(g3)} 帧 @ {FS:.0f} Hz)")

    cfg_p = PROJECT / "config.yaml"
    if not cfg_p.exists():
        cfg = read_config(str(TEMPLATE))
        cfg["project_name"] = PROJECT.name
        cfg["project_path"] = str(PROJECT)
        cfg["session_names"] = SESS
        cfg["keypoints"] = ["grav"]
        cfg["num_features"] = 3
        write_config(str(cfg_p), cfg)
        print(f"[cfg] -> {cfg_p}")
    cfg = read_config(str(cfg_p))

    from vame.model.create_training import create_trainset
    create_trainset(cfg, test_fraction=cfg["test_fraction"], split_mode="mode_2")
    print("[vame] train/test 序列已生成 → 可用 routeD_train.py 开训")


if __name__ == "__main__":
    main()
