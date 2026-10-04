"""配置加载与路径管理（全项目唯一的路径规则定义处）。"""

from pathlib import Path

import yaml

# 项目根目录 = utils.py 的上一级（Scripts_Mann/ 的父目录）
ROOT = Path(__file__).resolve().parent.parent


def load_config(config_name="config.yaml"):
    """读取项目根目录下的 yaml 配置，返回 dict。"""
    with open(ROOT / config_name, encoding="utf-8") as f:
        return yaml.safe_load(f)



def output_dir(cfg):
    """output 目录的绝对路径，不存在则创建。"""
    out_dir = ROOT / cfg["project"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def latent_path(cfg, rec_name):
    #录制文件对应的latent npz路径
    return output_dir(cfg) / f"{rec_name}_latent.npz"


def hmm_path(cfg, rec_name):
    #录制文件对应的hmm npz路径
    return output_dir(cfg) / f"{rec_name}_hmm.npz"  

def new_experiment_dir(cfg):
    """创建 output/exp_XXX/ 实验目录，编号自动递增。"""
    out = output_dir(cfg)
    existing = [p.name for p in out.glob("exp_*") if p.is_dir()]
    numbers = [int(n.split("_")[1]) for n in existing
               if n.split("_")[1].isdigit()]
    exp_dir = out / f"exp_{max(numbers, default=0) + 1:03d}"
    exp_dir.mkdir()
    return exp_dir


def latest_experiment_dir(cfg):
    """返回最新的实验目录（embed/可视化默认使用最近训练的模型）。"""
    out = output_dir(cfg)
    exps = sorted(p for p in out.glob("exp_*") if p.is_dir())
    if not exps:
        raise FileNotFoundError("没有实验目录，请先运行 train.py")
    return exps[-1]