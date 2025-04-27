"""
paths.py
========
Utility helpers for resolving project paths.

"""

from pathlib import Path
import yaml

# Root directory = two levels up from this file
ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT_DIR / "config" / "default.yml"
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"

def load_config(custom_cfg: Path | None = None) -> dict:
    """Load YAML configuration with override.

    Args:
        custom_cfg (Path | None): Path to user-supplied YAML file.

    Returns:
        dict: Merged configuration dictionary.
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if custom_cfg:
        with open(custom_cfg) as f:
            user = yaml.safe_load(f)
        cfg |= user  # Python 3.9 dict merge
    return cfg
