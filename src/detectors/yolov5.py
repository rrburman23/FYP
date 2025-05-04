from __future__ import annotations
from typing import List
import torch
from detectors.base import DetectorBase

class YOLOv5(DetectorBase):
    """
    YOLOv5-s detector via `torch.hub.load('ultralytics/yolov5')`.
    This class wraps the YOLOv5 model from the `ultralytics/yolov5` repository.
    """

    def __init__(self, model_name: str = "yolov5s", conf_thresh: float = 0.25, device: str | None = None):
        """
        Initialize YOLOv5 detector.

        Args:
            model_name (str): The version of YOLOv5 to use (e.g., 'yolov5s', 'yolov5m', etc.).
            conf_thresh (float): Confidence threshold for detections.
            device (str | None): Device to use for inference ("cuda" for GPU or "cpu" for CPU). Defaults to auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('ultralytics/yolov5', model_name, trust_repo=True).to(self.device).eval()
        self.model.conf = conf_thresh

    @torch.inference_mode()
    def detect(self, frame) -> List[list]:
        """
        Detect objects in the given frame using YOLOv5.

        Args:
            frame (np.ndarray): Input frame (image) to perform detection on.

        Returns:
            List[list]: Detected bounding boxes in the format [x1, y1, x2, y2, confidence].
        """
        result = self.model(frame, size=640)  # Run inference with input size of 640
        det = result.xyxy[0].cpu().numpy()   # Bounding boxes: x1 y1 x2 y2 conf cls
        return [[int(x1), int(y1), int(x2), int(y2), float(c)]
                for x1, y1, x2, y2, c, _ in det]
