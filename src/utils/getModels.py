"""
getModels.py
==================
Utility for downloading pretrained YOLOv3-416 model files (cfg, weights, labels).
Using https://pjreddie.com/darknet/yolo/
"""

import os
import urllib.request
from utils.paths import ROOT_DIR  # Assumes ROOT_DIR is a pathlib.Path object

# Remote URLs for YOLOv3-416 model files
AVAILABLE_MODELS = {
    "models/yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "models/yolov3.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights",
    "models/labels.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

def download_file(url: str, local_path: str) -> None:
    """Download a file if it doesn't exist."""
    try:
        print(f"[get_models] Downloading {os.path.basename(local_path)}...")
        urllib.request.urlretrieve(url, local_path)
        print(f"[get_models] Downloaded {os.path.basename(local_path)} ✔️")
    except Exception as e:
        print(f"[get_models] ❌ Failed to download {url}")
        print(e)

def get_models() -> None:
    """Download YOLOv3-416 model files if missing."""
    model_dir = ROOT_DIR / "models"
    os.makedirs(model_dir, exist_ok=True)

    for rel_path, url in AVAILABLE_MODELS.items():
        full_path = ROOT_DIR / rel_path
        if not full_path.exists():
            download_file(url, str(full_path))
        else:
            print(f"[get_models] {rel_path} already exists. Skipping ✅")

if __name__ == "__main__":
    get_models()
