"""
yolov5.py
─────────
Ultralytics YOLOv5 wrapper (hub load).
"""

from __future__ import annotations
from typing   import List
import numpy as np
import torch
from detectors.base import DetectorBase


class YOLOv5(DetectorBase):
    """
    YOLOv5-s detector via `torch.hub.load('ultralytics/yolov5')`.

    Returns:
        list[[x1,y1,x2,y2,conf], …] in pixel coordinates (ints).
    """

    def __init__(self,
                 model_name: str = "yolov5s",
                 conf_thresh: float = 0.25,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # trust_repo=True avoids hub prompt
        self.model = torch.hub.load('ultralytics/yolov5', model_name,
                                    trust_repo=True).to(self.device).eval()
        self.model.conf = conf_thresh

    @torch.inference_mode()
    def detect(self, frame) -> List[list]:
        # model handles RGB/BGR correctly – pass as‑is
        result = self.model(frame, size=640)
        det = result.xyxy[0].cpu().numpy()          # x1 y1 x2 y2 conf cls
        return [[int(x1), int(y1), int(x2), int(y2), float(c)]
                for x1, y1, x2, y2, c, _ in det]
