# -*- coding: utf-8 -*-

'''
模型数据导入


'''

from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


class Session:
    """单条记录的只读视图（25 Hz）。"""

    def __init__(self, name):
        self.name = name
        p = Path(CFG["data"]["dir"]) / f"{name}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        d = np.load(p)
        self.raw6 = d["raw6"]            # (T,6)
        self.feat9 = d["feat9"]          # (T,9)
        self.t = d["t"]                  # (T,)
        self.interp_mask = d["interp_mask"] if "interp_mask" in d.files else None
        self.fs = float(CFG["data"]["fs"])

    @property
    def T(self):
        return len(self.t)

    @property
    def grav(self):
        """重力方向 3 通道（feat9 3:6，单位向量，原尺度）。"""
        return self.feat9[:, 3:6]

    @property
    def dyn6(self):
        """动态 6 通道（linacc3 + gyro3，已滚动 z 归一化）。"""
        return np.concatenate([self.feat9[:, 0:3], self.feat9[:, 6:9]], axis=1)


def sessions(names=None):
    """全部（或指定）session 的 Session 列表。"""
    names = names or CFG["data"]["sessions"]
    return [Session(n) for n in names]


if __name__ == "__main__":
    # 自检：长度、NaN、fs、字段
    for s in sessions():
        nan = np.isnan(s.feat9).sum() + np.isnan(s.raw6).sum()
        dt = np.diff(s.t)
        ok = (abs(dt - 1 / s.fs) < 1e-6).all() if len(dt) else False
        print(f"{s.name}: T={s.T} fs_ok={ok} nan={nan} "
              f"interp={int(s.interp_mask.sum()) if s.interp_mask is not None else '?'}")
