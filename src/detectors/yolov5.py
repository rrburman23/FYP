from __future__ import annotations
from typing import List
import torch
import cv2
import numpy as np
from detectors.base import DetectorBase


class YOLOv5(DetectorBase):
    """
    YOLOv5 object detector implementation using torch.hub.

    Inherits from DetectorBase and implements required abstract methods.
    Handles BGR→RGB conversion and memory contiguity for OpenCV compatibility.

    Attributes:
        device (str): Computation device (cuda/cpu)
        model (torch.nn.Module): Loaded YOLOv5 model
        conf_thresh (float): Confidence threshold for detections
    """

    def __init__(
        self,
        model_name: str = "yolov5s",
        conf_thresh: float = 0.4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize YOLOv5 detector with specified parameters.

        Args:
            model_name: YOLOv5 model variant (n/s/m/l/x)
            conf_thresh: Minimum confidence threshold (0-1)
            device: Force computation device ('cuda'/'cpu'), auto-detects if None
        """
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_thresh = conf_thresh

        # Load model from Torch Hub with silent mode (no progress bars)
        self.model = torch.hub.load(
            "ultralytics/yolov5", model_name, trust_repo=True, verbose=False
        )
        self.model.conf = conf_thresh  # Set confidence threshold
        self.model.to(self.device).eval()  # Set model to evaluation mode

    @torch.inference_mode()
    def detect(self, frame: np.ndarray) -> List[List[float]]:
        """
        Perform object detection on input frame.

        Args:
            frame: Input frame in BGR format (OpenCV default)

        Returns:
            List of detections in format [x1, y1, x2, y2, confidence]
        """
        # Convert BGR to RGB and ensure memory contiguity
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_contig = np.ascontiguousarray(frame_rgb)

        # Perform inference with fixed input size (faster processing)
        results = self.model(frame_contig, size=640, augment=False)

        # Extract and format detections
        return self._parse_detections(results)

    def _parse_detections(self, results) -> List[List[float]]:
        """
        Parse raw model outputs into standardized detection format.

        Args:
            results: Raw output from YOLOv5 model

        Returns:
            Filtered detections as list of [x1, y1, x2, y2, confidence]
        """
        detections = []
        # results.xyxy[0] contains [x1, y1, x2, y2, confidence, class]
        for det in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = det
            if conf >= self.conf_thresh:
                detections.append(
                    [
                        int(x1),  # Convert to int for pixel coordinates
                        int(y1),
                        int(x2),
                        int(y2),
                        float(conf),  # Confidence as float for metrics
                    ]
                )
        return detections
