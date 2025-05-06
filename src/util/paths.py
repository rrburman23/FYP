"""
paths.py
========
Path resolution and configuration loader for the project.
"""

from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

# Define project root and key directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "data"
TEMP_DIR = OUTPUT_DIR / "temp"

def load_config(config_file: str = "default.yml") -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        config_file (str): Name of the configuration file (default: 'default.yml')
    
    Returns:
        dict: Parsed configuration dictionary
    
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    config_path = CONFIG_DIR / config_file
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            logger.error(f"Configuration file is empty: {config_path}")
            raise ValueError(f"Configuration file is empty: {config_path}")
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse configuration file {config_path}: {e}")
        raise yaml.YAMLError(f"Failed to parse configuration file {config_path}: {e}")