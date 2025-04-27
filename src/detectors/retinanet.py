"""
retinanet.py - TorchVision RetinaNet wrapper.
"""

from __future__ import annotations
from typing import List
import numpy as np, torch
from torchvision import models, transforms as T
from detectors.base import DetectorBase


class RetinaNet(DetectorBase):
    def __init__(self, conf_thresh: float = 0.4, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = models.detection.retinanet_resnet50_fpn(weights="DEFAULT")\
                         .to(self.device).eval()
        self.tf   = T.Compose([T.ToTensor()])
        self.th   = conf_thresh

    @torch.inference_mode()
    def detect(self, frame) -> List[list]:
        out = self.model([self.tf(frame[:,:,::-1]).to(self.device)])[0]
        boxes, scores = out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()
        return [[int(x1), int(y1), int(x2), int(y2), float(s)]
                for (x1,y1,x2,y2), s in zip(boxes, scores) if s > self.th]
