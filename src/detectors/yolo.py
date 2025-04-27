"""
yolo.py
───────────────
OpenCV-DNN wrapper for custom YOLOv3 cfg / weights.
"""

from __future__ import annotations
from typing import List
from pathlib import Path
import cv2, numpy as np
from detectors.base import DetectorBase
from utils.paths import ROOT_DIR, load_config


class YOLOv3(DetectorBase):
    """Custom YOLO v3 (Darknet) inference via OpenCV DNN."""

    def __init__(self,
                 conf_thresh: float = 0.35,
                 nms_thresh: float = 0.45):
        ycfg = load_config()["yolo"]  # paths in default.yml
        cfg     = ROOT_DIR / ycfg["cfg"]
        weights = ROOT_DIR / ycfg["weights"]
        for p in (cfg, weights):
            if not p.exists():
                raise FileNotFoundError(p)

        self.net = cv2.dnn.readNet(str(weights), str(cfg))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_thresh, self.nms_thresh = conf_thresh, nms_thresh
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    # ------------------------------------------------------------------ #
    def detect(self, frame) -> List[list]:
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), swapRB=True)
        self.net.setInput(blob)
        layer_outs = self.net.forward(self.out_layers)

        boxes, confs = [], []
        for out in layer_outs:
            for det in out:
                scores = det[5:]; conf = scores.max()
                if conf < self.conf_thresh:
                    continue
                cx, cy, bw, bh = det[:4] * [W, H, W, H]
                x1, y1 = int(cx - bw/2), int(cy - bh/2)
                boxes.append([x1, y1, int(bw), int(bh)])
                confs.append(float(conf))

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)
        idxs = np.array(idxs).reshape(-1)       # guarantees 1‑D ndarray
        res  = []
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            res.append([x, y, x+bw, y+bh, confs[i]])
        print(f"[YOLOv3] frame detections: {len(res)}")
        return res
