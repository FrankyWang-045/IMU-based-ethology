from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(name="config.yaml"):
    with open(ROOT / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_dir(cfg):
    out = Path(cfg["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out