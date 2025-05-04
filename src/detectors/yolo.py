from __future__ import annotations
from typing import List
import cv2
import numpy as np
from detectors.base import DetectorBase
from utils.paths import ROOT_DIR, load_config
from utils.getModels import get_models

class YOLOv3(DetectorBase):
    """
    Custom YOLOv3 (Darknet) detector using OpenCV DNN backend.
    This class reads a YOLOv3 configuration and weights file and performs detection using OpenCV.
    """

    def __init__(self, conf_thresh: float = 0.35, nms_thresh: float = 0.45):
        """
        Initialize YOLOv3 detector.

        Args:
            conf_thresh (float): Confidence threshold for detections.
            nms_thresh (float): Non-maximum suppression threshold to avoid overlapping boxes.
        """
        get_models()
        ycfg = load_config()["yolo"]  # Load YOLO config
        cfg = ROOT_DIR / ycfg["cfg"]
        weights = ROOT_DIR / ycfg["weights"]
        
        for p in (cfg, weights):
            if not p.exists():
                raise FileNotFoundError(p)

        # Load YOLOv3 model using OpenCV DNN
        self.net = cv2.dnn.readNet(str(weights), str(cfg))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        ln = self.net.getLayerNames()
        self.out_layers = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame) -> List[list]:
        """
        Perform object detection using YOLOv3.

        Args:
            frame (np.ndarray): Input frame (image) to detect objects.

        Returns:
            List[list]: Detected bounding boxes in the format [x1, y1, x2, y2, confidence].
        """
        H, W = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        self.net.setInput(blob)
        layer_outs = self.net.forward(self.out_layers)

        boxes, confs = [], []
        for out in layer_outs:
            for det in out:
                scores = det[5:]
                conf = scores.max()
                if conf < self.conf_thresh:
                    continue
                cx, cy, bw, bh = det[:4] * [W, H, W, H]
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                boxes.append([x1, y1, int(x2), int(y2)])
                confs.append(float(conf))

        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf_thresh, self.nms_thresh)
        idxs = np.array(idxs).reshape(-1)  # Flatten the indices array
        res = []
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            res.append([x, y, x + bw, y + bh, confs[i]])

        return res
