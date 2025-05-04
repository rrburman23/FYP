"""
paths.py
========
Path resolution and configuration loader for the project.
"""

from pathlib import Path
import yaml
from typing import Optional, Dict

# Root directory: assumed to be 2 levels up from this file
ROOT_DIR = Path(__file__).resolve().parents[2]

# Commonly used paths
CONFIG_PATH = ROOT_DIR / "config" / "default.yml"
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "outputs"


def load_config(custom_cfg: Optional[Path] = None) -> Dict:
    """
    Load project configuration from YAML file, with optional override.

    Args:
        custom_cfg (Path | None): Optional path to a custom config file.

    Returns:
        dict: Configuration dictionary with values from base and override.
    """
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    
    if custom_cfg:
        with open(custom_cfg) as f:
            override_cfg = yaml.safe_load(f)
        cfg |= override_cfg  # Python 3.9+ dict merge

    return cfg
